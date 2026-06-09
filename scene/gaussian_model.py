#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
# 基于原始 3D 高斯渲染（3DGS）的扩展版本，核心新增了高斯掩码（Mask）功能，
# 允许对每个 3D 高斯进行 “保留 / 剔除” 的二分类决策。
# 通过引入掩码参数和对应的训练、剪枝逻辑，实现了对场景中特定区域的高斯精细化控制（如剔除无关区域高斯、保留目标区域细节）

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from icecream import ic
from copy import deepcopy
class GaussianModel:
    # ----------------------------- Setup -----------------------------
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        # [MODIFIED] 使用 Softmax 而非 LogSoftmax，使梯度幅度正常
        #self.mask_activation = nn.Softmax(dim=-1)
        self.mask_activation = nn.LogSoftmax(dim=-1)

        self.rotation_activation = torch.nn.functional.normalize

    # ----------------------------- Init -----------------------------
    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask_score = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        #self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    # ----------------------------- State save/restore -----------------------------
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._mask_score,
            self.max_radii2D,
            self.xyz_gradient_accum,
            #self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._mask_score,
            self.max_radii2D,
            xyz_gradient_accum,
            #xyz_gradient_accum_abs,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
       # self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def restore_from_3dgs(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        mask_scores = torch.ones((self._xyz.shape[0], 2), dtype=torch.float, device="cuda") @ torch.tensor(
            [[10.0, 0.0], [0.0, 1.0]], dtype=torch.float, device="cuda"
        )
        self._mask_score = nn.Parameter(mask_scores.requires_grad_(True))
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
       # self.xyz_gradient_accum_abs = torch.zeros((self.xyz_gradient_accum.shape[0], 1), device="cuda")
        self.denom = denom

        state_dict = deepcopy(opt_dict)
        state_dict["state"][6] = state_dict["state"][5]
        state_dict["state"][5] = state_dict["state"][4]
        state_dict["state"][4] = {
            "step": state_dict["state"][0]["step"],
            "exp_avg": torch.zeros_like(self._mask_score),
            "exp_avg_sq": torch.zeros_like(self._mask_score),
        }
        tmp_param_group = deepcopy(state_dict["param_groups"][3])
        tmp_param_group["lr"] = training_args.mask_lr
        tmp_param_group["name"] = "mask"
        tmp_param_group["params"] = [4]
        state_dict["param_groups"].insert(4, tmp_param_group)
        state_dict["param_groups"][5]["params"] = [5]
        state_dict["param_groups"][6]["params"] = [6]
        self.optimizer.load_state_dict(state_dict)

    # ----------------------------- Property Accessors -----------------------------
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)
    @property
    def num_primitives(self):
        return self._xyz.shape[0]  # 高斯数量=位置张量的行数
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    # [MODIFIED] Mask 输出: 支持 annealing 硬采样
    #@property
    #def get_mask(self):
        #hard_flag = getattr(self, "current_iter", 1001) > 1000
        #return nn.functional.gumbel_softmax(
            #self.mask_activation(self._mask_score), hard=hard_flag
        #)[:, 0:1]
    @property
    def get_mask(self):
        return nn.functional.gumbel_softmax(self.mask_activation(self._mask_score), hard=True)[:, 0:1]

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # ----------------------------- Initialization -----------------------------
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialization : ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        dist2 = torch.sqrt(dist2)
        scales = torch.log(dist2)[...,None].repeat(1, 3)  # 缩放初始值（log确保exp后为正）
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1  # 初始旋转为单位四元数（无旋转）

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        mask_scores = torch.ones((fused_point_cloud.shape[0], 2), dtype=torch.float, device="cuda") @ torch.tensor(
            [[10.0, 0.0], [0.0, 1.0]], dtype=torch.float, device="cuda"
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._mask_score = nn.Parameter(mask_scores.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # ----------------------------- Training Setup -----------------------------
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
       # self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._mask_score], "lr": training_args.mask_lr, "name": "mask"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    # ----------------------------- AbsGS Integration -----------------------------
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
     
        #grad = viewspace_point_tensor.grad[update_filter]
       # grad_norm = torch.norm(grad, dim=-1, keepdim=True)
       # self.xyz_gradient_accum[update_filter] += grad_norm
        #self.denom[update_filter] += 1
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom

        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)



        prune_mask = (self.get_opacity < min_opacity).squeeze()
        sample_times = 10
        batched_mask_score = self._mask_score.unsqueeze(0).repeat(sample_times, 1, 1)
        res = nn.functional.gumbel_softmax(self.mask_activation(batched_mask_score), hard=True)[:, :, 0].sum(0)
        prune_mask = torch.logical_or(prune_mask, res == 0.0)

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)


        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._mask_score.shape[1]):
            l.append('masks_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def construct_list_of_attributes_default(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
# 保存包含掩码的模型
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        masks = self._mask_score.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, masks, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
#  保存仅目标区域的模型（save_ply_default方法）新增方法，仅保存掩码为 “保留” 的高斯，生成兼容原始 3DGS 的 PLY 文件  
    def save_ply_default(self, path):
        mkdir_p(os.path.dirname(path))
        mask = self.get_mask
        mask = mask.to(torch.bool).squeeze(1)
        # self._xyz = self._xyz[mask]
        # self._features_dc = self._features_dc[mask]
        # self._features_rest = self._features_rest[mask]
        # self._opacity = self._opacity[mask]
        # self._scaling = self._scaling[mask]
        # self._rotation = self._rotation[mask]

        xyz = self._xyz[mask].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity[mask].detach().cpu().numpy()
        scale = self._scaling[mask].detach().cpu().numpy()
        rotation = self._rotation[mask].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_default()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        ic(f"Saving Gaussians. Number of Gaussians: {self._xyz[mask].shape[0]}")

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
  
# 加载包含掩码的模型
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        masks = np.zeros((xyz.shape[0], 2))
        masks[:, 0] = np.asarray(plydata.elements[0]["masks_0"])
        masks[:, 1] = np.asarray(plydata.elements[0]["masks_1"])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._mask_score = nn.Parameter(torch.tensor(masks, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask, store_grads=False):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                if store_grads:
                    grad = group["params"][0].grad[mask]  # 保留裁剪后的梯度
                # 裁剪参数并重新封装为可学习参数
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                if store_grads:
                    group["params"][0].grad = grad  # 恢复裁剪后的梯度
                self.optimizer.state[group['params'][0]] = stored_state  # 恢复优化器状态

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def mercy_points(self, densification_statistics_dict, lambda_mercy=2, mercy_minimum=2, mercy_type='redundancy_opacity'):
        # 计算高斯的空间冗余分数的均值和标准差（self._splatted_num_accum存储冗余分数）
        mean = self._splatted_num_accum.squeeze().float().mean(dim=0, keepdim=True)
        std =  self._splatted_num_accum.squeeze().float().var(dim=0, keepdim=True).sqrt()

        # 计算剪枝阈值：max(均值+lambda_mercy*标准差, 最小冗余分数)，对应论文4.1节τ_p = max(μ+λ_rσ, 3)
        threshold = max((mean + lambda_mercy*std).item(), mercy_minimum)

        # 筛选冗余高斯：冗余分数>阈值的高斯标记为待剪枝（mask=True表示待剪枝）
        mask = (self._splatted_num_accum > threshold).squeeze()

        if mercy_type == 'redundancy_opacity':
            # 策略1：联合冗余分数和透明度剪枝
            # 待剪枝高斯中，仅剪枝透明度低于中位数的高斯（保留高透明度的关键高斯）
            mask[mask.clone()] = self.get_opacity[mask].squeeze() < self.get_opacity[mask].median()
        elif mercy_type == 'redundancy_random':
            # 策略2：冗余高斯中随机剪枝50%
            mask[mask.clone()] = torch.rand(mask[mask].shape, device="cuda").squeeze() < 0.5
        elif mercy_type == 'opacity':
            # 策略3：仅基于透明度剪枝
            threshold = self.get_opacity.quantile(0.045)  # 透明度分位数阈值
            mask = (self.get_opacity < threshold).squeeze()
        elif mercy_type == 'redundancy_opacity_opacity':
            # 策略4：联合冗余+透明度、单独透明度剪枝（增强剪枝效果）
            mask[mask.clone()] = self.get_opacity[mask].squeeze() < self.get_opacity[mask].median()
            threshold = torch.min(self.get_opacity.quantile(0.03), torch.tensor([0.05], device="cuda"))
            mask = torch.logical_or(mask, (self.get_opacity < threshold).squeeze())
            
        # 执行剪枝（删除mask标记的高斯）
        self.prune_points(mask)
        densification_statistics_dict["n_points_mercied"] = mask.sum()
        densification_statistics_dict["redundancy_threshold"] = mean + lambda_mercy*std
        densification_statistics_dict["opacity_threshold"] = threshold if mercy_type in ['redundancy_opacity_opacity', 'opacity'] else 0
    
    def prune(self, min_opacity, extent, max_screen_size, densification_statistics_dict, store_grads=False):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        densification_statistics_dict["n_points_pruned"] = prune_mask.sum()
        self.prune_points(prune_mask, store_grads=store_grads)
    
    def prune_points(self, mask, store_grads=False):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask, store_grads=store_grads)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._mask_score = optimizable_tensors["mask"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
       # self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        #self.max_weight = self.max_weight[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_masks, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "mask": new_masks,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._mask_score = optimizable_tensors["mask"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
       # self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
       # max_weight = torch.zeros((new_xyz.shape[0]), device="cuda")
       # self.max_weight = torch.cat((self.max_weight,max_weight),dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_mask = self._mask_score[selected_pts_mask].repeat(N,1)  #

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_mask, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
# 原始 3DGS 的密度化（分裂 / 克隆）仅复制位置、颜色等参数，扩展版本新增掩码复制，确保新生成的高斯继承原高斯的掩码属性
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_masks = self._mask_score[selected_pts_mask] #
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_masks, new_scaling, new_rotation)

#  单独掩码剪枝（支持仅基于掩码进行剪枝（如推理时快速剔除背景高斯）
    def mask_prune(self):
        sample_times = 10
        batched_mask_score = self._mask_score.unsqueeze(0).repeat(sample_times, 1, 1)
        res = nn.functional.gumbel_softmax(self.mask_activation(batched_mask_score), hard=True)[:, :, 0].sum(0)
        prune_mask = (res==0.).squeeze()
        # prune_mask = (torch.sigmoid(self._mask_score[:, 1:2]) <= 0.01).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
