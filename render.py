#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
# 3D 高斯 Splatting模型的专用渲染脚本，
# 其核心功能是加载训练完成的 3D 高斯模型，对指定场景的训练集或测试集相机视角进行批量渲染，生成可视化结果并保存真实图像（GT）用于后续质量评估

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_orig_3dgs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model_3dgs import GaussianModel
#from scene.gaussian_model import GaussianModel
import pandas as pd
#from gaussian_renderer import render

def measure_fps(iteration, views, gaussians, pipeline, background):
    total_time = 0.0
    for _, view in enumerate(views):
        render_orig_3dgs(view, gaussians, pipeline, background, scaling_modifier=1.0, override_color=None,measure_fps=False)
    for _, view in enumerate(views):
        #result = render(view, gaussians, pipeline, background, scaling_modifier=1.0, override_color=None,measure_fps=True)
        result = render_orig_3dgs(view, gaussians, pipeline, background, scaling_modifier=1.0, override_color=None, measure_fps=True)
        frame_time = 1.0 / result["FPS"]  # 单帧渲染时间（秒）
        total_time += frame_time
        

    avg_fps = len(views) / total_time 

    return pd.Series([avg_fps], index=["FPS"], name=f"iteration_{iteration}")

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render_orig_3dgs(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,skip_measure_fps : bool):
    with torch.no_grad():
        # 初始化场景与高斯模型
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # 背景颜色
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 渲染配置
        configurations = {}
        if not skip_train:
            configurations["train"] = scene.getTrainCameras()
        if not skip_test:
            configurations["test"] = scene.getTestCameras()

        # 初始化 DataFrame 用于存储帧率结果
        df = pd.DataFrame()

        # 🔹 直接加载当前迭代下的默认点云
        ply_path = os.path.join(
            scene.model_path,
            "point_cloud",
            f"iteration_{scene.loaded_iter}",
            "point_cloud.ply",
        )
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"Cannot find point cloud file: {ply_path}")

        scene.gaussians.load_ply(ply_path)

        # 渲染训练 / 测试集
        for split_name, cameras in configurations.items():
            render_set(
                dataset.model_path,
                split_name,
                scene.loaded_iter,
                cameras,
                gaussians,
                pipeline,
                background, 
            )        
        
        if not skip_measure_fps:
                # 追加当前模型的帧率到数据框
            fps_df = measure_fps(
                scene.loaded_iter,
                scene.getTrainCameras() + scene.getTestCameras(),
                gaussians,
                pipeline,
                background
            )

            df = pd.concat([df, fps_df.to_frame().T], ignore_index=True)


        # 步骤3：所有模型处理完毕后，统一保存所有帧率数据
        if not skip_measure_fps and not df.empty:
            with open(os.path.join(dataset.model_path, f"fps_results.json"), 'w') as f:
                f.write(df.T.to_json())
        #if fps_df is not None and not fps_df.empty: 
            #fps_path = os.path.join(dataset.model_path, "fps_results.json") 
            #with open(fps_path, "w") as f: 
                #f.write(fps_df.T.to_json()) 
            #print(f"[INFO] FPS results saved to {fps_path}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_measure_fps", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,args.skip_measure_fps)
