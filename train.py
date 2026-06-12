#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Modified by Kuang (2025)
# - Integrated AbsGS absolute gradient densification criterion
# - Retained MaskGaussian pruning and mask learning mechanism
#

import os
import torch
import time
from random import randint
from utils.loss_utils import l1_loss,ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
torch.cuda.empty_cache()
max_gpu_mem = 0          # max_memory_reserved
max_gpu_alloc = 0        # max_memory_allocated  👈 加这个
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


simple_scheduler = lambda value, iter, start, end: value if start <= iter <= end else 0
dens_statistic_dict = {"n_points_cloned": 0, "n_points_split": 0, "n_points_mercied": 0, "n_points_pruned": 0, "redundancy_threshold": 0, "opacity_threshold": 0}

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    gaussians._splatted_num_accum = torch.zeros((gaussians.num_primitives, 1), device="cuda")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    #fine_tune_start = opt.iterations - 3000 if args.mercy_points else opt.iterations
    total_train_start = time.time() 
    for iteration in range(first_iter, opt.iterations + 1):
        gaussians.current_iter = iteration

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                with torch.no_grad():
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview(
                            (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                        )
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
            except Exception:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # 每 1000 次提升一次 SH 阶数
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 选取随机相机
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # ----------------------------- 渲染与损失 -----------------------------
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii, mask = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            render_pkg["mask"],
        )

        # 计算透明度稀疏性损失（辅助剪枝筛选低透明度高斯）
        if args.lambda_alpha_regul == 0:
            Lalpha_regul = torch.tensor([0.], device=image.device)
        else:
            # 仅对可见高斯计算透明度L₁损失，鼓励透明度降低
            points_opacity = gaussians.get_opacity[visibility_filter]
            Lalpha_regul = points_opacity.abs().mean()

        # 计算总损失：L1损失 + SSIM损失 + 透明度稀疏性损失
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        Lssim = 1.0 - ssim(image, gt_image)
        mask_loss = torch.mean(mask * (1 - mask))
        lambda_mask = simple_scheduler(opt.lambda_mask, iteration, opt.mask_from_iter, opt.mask_until_iter)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim + Lalpha_regul * args.lambda_alpha_regul+ lambda_mask * mask_loss
        
        #loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + lambda_mask * mask_loss


        loss.backward()  # 反向传播计算梯度
        iter_end.record()

        # ====================== 每轮更新显存峰值 ======================
        global max_gpu_mem
        global max_gpu_alloc
        # 🔥 这是【真实、完整、全部】显存占用（和 nvidia-smi 一样）
        current_mem = torch.cuda.max_memory_reserved() / 1024**3
        current_alloc = torch.cuda.max_memory_allocated() / 1024**3  # 👈 加这个
        if current_mem > max_gpu_mem:
            max_gpu_mem = current_mem
        if current_alloc > max_gpu_alloc:
            max_gpu_alloc = current_alloc  # 👈 加这个

        # 重置峰值统计，下一帧重新计算
        torch.cuda.reset_peak_memory_stats()
        # ==============================================================

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            num_used_gs = int(mask.detach().sum())

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "num_used_gs": num_used_gs})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # ----------------------------- 日志 & 保存 -----------------------------
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                num_used_gs,
            )
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # ----------------------------- Densify & Prune -----------------------------
            if iteration < opt.densify_until_iter:
                # 记录 radii
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration % (opt.densification_interval // 2) == 0:
                    mask_backup = gaussians._mask_score.data.clone()
                    gaussians._mask_score.requires_grad_(False)
                    gaussians._mask_score.data.fill_(10.0)
                    render_pkg_abs = render(viewpoint_cam, gaussians, pipe, bg)
                    if render_pkg_abs["viewspace_points"].grad is not None:
                        gaussians.add_densification_stats(
                            render_pkg_abs["viewspace_points"],
                            torch.ones_like(visibility_filter, dtype=torch.bool),
                        )
                    gaussians._mask_score.data = mask_backup
                    gaussians._mask_score.requires_grad_(True)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # [Freeze Mask During Densify]
                    for p in [gaussians._mask_score]:
                        p.requires_grad_(False)
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold,0.005,scene.cameras_extent,size_threshold,)
                    for p in [gaussians._mask_score]:
                        p.requires_grad_(True)

                if (iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()
            elif args.prune_dead_points and iteration % opt.densification_interval == 0:
                gaussians.prune(1/255, scene.cameras_extent, None, dens_statistic_dict)

            if args.mercy_points and iteration % args.mercy_interval == 0 and iteration >= opt.densify_until_iter: #and iteration <= fine_tune_start:
                with torch.no_grad():
                 
                    gaussians._splatted_num_accum, _ = scene.calculate_redundancy_metric(pixel_scale=args.box_size)
                    gaussians._splatted_num_accum = gaussians._splatted_num_accum.unsqueeze(1)
                    gaussians.mercy_points(dens_statistic_dict,
                               lambda_mercy=args.lambda_mercy,
                               mercy_minimum=args.mercy_minimum,
                               mercy_type=args.mercy_type)
                    
                    #print(f"[ITER {iteration}] Pruned {dens_statistic_dict['n_points_mercied'].item()} redundant Gaussians "
      #f"(τ_r={dens_statistic_dict['redundancy_threshold'].item():.3f}, "
     # f"τ_α={dens_statistic_dict['opacity_threshold'].item():.3f}).")

            # [Mask Timing Adjust]
            #if iteration > opt.densify_until_iter and iteration % opt.mask_prune_iter == 0:
            if iteration > opt.densify_from_iter and iteration % (2 * opt.mask_prune_iter) == 0:
                print(f"[ITER {iteration}] Running mask_prune() ...")
                gaussians.mask_prune()

            # ----------------------------- 优化器步进 -----------------------------
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")
    total_train_end = time.time()
    total_duration = total_train_end - total_train_start
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    print(f"\n=====================================")
    print(f"Total training time: {hours}h {minutes}m {seconds}s")
    print(f"Total training time (precise): {total_duration:.2f} seconds")
    print(f"=====================================")
    print(f"✅ PEAK GPU MEMORY RESERVED:  {max_gpu_mem:.2f} GB")
    print(f"✅ PEAK GPU MEMORY ALLOCATED: {max_gpu_alloc:.2f} GB")  # 👈 加这个
    print("=======================================================\n")

# ----------------------------- 其他函数保持原样 -----------------------------
def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID", str(uuid.uuid4()))
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder:", args.model_path)
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None
    if not TENSORBOARD_FOUND:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, num_used_gs=-1):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        tb_writer.add_scalar("num_used_gs", num_used_gs, iteration)
        tb_writer.add_scalar("GPU_mem", torch.cuda.memory_reserved() / 1024**3, iteration)
        tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        if iteration == testing_iterations[-1]:
                            tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render", image[None], iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}/ground_truth", gt_image[None], iteration
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[i * 1000 for i in range(1, 31)],
    )
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing", args.model_path)
    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    print("\nTraining complete.")
