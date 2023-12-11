from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = '0.1.0'

def get_extensions():
    extensions = []
    try:
        if torch.cuda.is_available():
            ext_ops = CUDAExtension('smplx_mpr.cuda.rasterizer', [
            'smplx_mpr/cuda/rasterizer.cpp',  # noqa: E501
            'smplx_mpr/cuda/rasterizer_kernel.cu',  # noqa: E501
            ])
            extensions.append(ext_ops)
    except Exception as e:
        raise RuntimeError
    return extensions

setup(
    name='smplx_mpr',
    version=__version__,
    ext_modules=get_extensions(),
    cmdclass = {'build_ext': BuildExtension},
    packages=find_packages(),
)
