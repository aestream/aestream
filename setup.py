from setuptools import setup
import glob
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="aedat",
    ext_modules=[CppExtension("aedat", ["convert.cpp",], libraries=["lz4"]),],
    cmdclass={"build_ext": BuildExtension},
)
