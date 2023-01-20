import os

from skbuild import setup  # Use scikit-build

# import torch
# from torch.utils import cpp_extension

pwd = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(pwd, "README.md"), encoding="utf-8") as fp:
    readme_text = fp.read()

# C++ config
cmake_args = [
    "-DWITH_PYTHON=1",
    # f"-DCMAKE_PREFIX_PATH='{os.path.dirname(torch.__file__)};{torch.utils.cmake_prefix_path}'",
    # f"-DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI={1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0} -fPIC'",
]

# Define extension based on CUDA availability
# cuda_home = cpp_extension._find_cuda_home()
# if cuda_home is not None:
#     archs = ";".join([x[3:] for x in torch.cuda.get_arch_list()])
#     flags = " ".join(cpp_extension._get_cuda_arch_flags())
#     cmake_args += [
#         #f"-DCMAKE_CUDA_ARCHITECTURES='{archs}'",
#         "-DCMAKE_CUDA_ARCHITECTURES='50;60;70;75;80;86'",
#         f'-DCMAKE_CUDA_FLAGS={flags}',
#         f"-DCMAKE_CUDA_COMPILER={cuda_home}/bin/nvcc",
#         f"-DCUDA_INCLUDE_DIRS={cuda_home}/include",
#     ]
#     print(flags)

# Setuptools entrypoint
setup(
    name="aestream",
    version="0.4.0",
    author="Jens E. Pedersen, Christian Pehle",
    author_email="jens@jepedersen.dk, christian.pehle@gmail.com",
    url="https://github.com/norse/aestream",
    description="Streaming library for Address-Event Representation (AER) data",
    license="MIT",
    long_description=readme_text,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    packages=["aestream"],
    install_requires=["numpy"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Software Development :: Libraries",
        "Topic :: System :: Hardware :: Universal Serial Bus (USB)",
    ],
    cmake_args=cmake_args,
)
