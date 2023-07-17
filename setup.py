import os
from skbuild import setup

pwd = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(pwd, "README.md"), encoding="utf-8") as fp:
    readme_text = fp.read()

# C++ config
cmake_args = ["-DUSE_PYTHON=1"]

# Activate CUDA, if found
try:
    import torch
    from torch.utils import cpp_extension

    cuda_home = cpp_extension._find_cuda_home()
    if cuda_home is not None:
        flags = " ".join(cpp_extension._get_cuda_arch_flags())
        cmake_args += [
            f"-DUSE_CUDA=1",
            f"-DCMAKE_CUDA_FLAGS={flags}",
            f"-DCMAKE_CUDA_COMPILER={cuda_home}/bin/nvcc",
            f"-DCUDA_INCLUDE_DIRS={cuda_home}/include",
        ]
except:
    pass

# Setuptools entrypoint
setup(
    name="aestream",
    version="0.6.0",
    author="Jens E. Pedersen",
    author_email="jens@jepedersen.dk",
    url="https://github.com/aestream/aestream",
    description="Streaming library for Address-Event Representation (AER) data",
    license="MIT",
    long_description=readme_text,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    packages=["aestream"],
    install_requires=["numpy", "pysdl2-dll"],
    extras_require={"torch": ["torch"]},
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Software Development :: Libraries",
        "Topic :: System :: Hardware :: Universal Serial Bus (USB)",
    ],
    cmake_args=cmake_args,
    package_data={"aestream": ["*.pyi", "*.typed"]},
)
