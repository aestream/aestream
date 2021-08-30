from setuptools import setup
import glob
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="DVSData",
    ext_modules=[CppExtension("dvs2tensor", ["dvs2tensor.cpp"], libraries=["lz4", "caer"]),],
    cmdclass={"build_ext": BuildExtension},
)