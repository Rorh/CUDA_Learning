# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="lade_cuda",
    ext_modules=[
        CUDAExtension(
            name="lade_cuda",
            sources=["lade_cuda.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
