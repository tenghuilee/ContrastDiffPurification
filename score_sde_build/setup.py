from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import torch

if torch.cuda.is_available():
    ext_modules=[
        CUDAExtension('score_sde.op.fused', [
            'score_sde/op/fused_bias_act.cpp',
            'score_sde/op/fused_bias_act_kernel.cu',
        ]),
        CUDAExtension('score_sde.op.upfirdn2d_op', [
            'score_sde/op/upfirdn2d.cpp',
            'score_sde/op/upfirdn2d_kernel.cu',
        ]),
    ]
else:
    ext_modules=[
        CppExtension('score_sde.op.fused', [
            'score_sde/op/fused_bias_act.cpp',
        ]),
        CppExtension('score_sde.op.upfirdn2d_op', [
            'score_sde/op/upfirdn2d.cpp',
        ]),
    ]

setup(
    name='score_sde',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension,
    },
    author="https://github.com/yang-song/score_sde_pytorch.git",
    description='score-sde adopt from https://github.com/yang-song/score_sde_pytorch.git. The setup.py is added for better access.',
)

