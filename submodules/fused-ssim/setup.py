import sys
# 替换为你自己的 torch 安装路径（pip show torch 查到的 Location）
sys.path.append("/home/sunfanshu/miniconda3/envs/maskgs/lib/python3.9/site-packages")
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="fused_ssim",
    packages=['fused_ssim'],
    ext_modules=[
        CUDAExtension(
            name="fused_ssim_cuda",
            sources=[
            "ssim.cu",
            "ext.cpp"])
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
