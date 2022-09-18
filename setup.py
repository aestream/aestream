from os import path
from setuptools import setup
import torch
from torch.utils import cpp_extension

pwd = path.abspath(path.dirname(__file__))
with open(path.join(pwd, "README.md"), encoding="utf-8") as fp:
    readme_text = fp.read()

with open(path.join(pwd, "requirements.txt"), encoding="utf-8") as fp:
    install_requires = fp.read()

# Setup C++ sources
cpp_sources = [
    # Module code
    "src/pybind/module.cpp",
    # UDP
    "src/pybind/udp.cpp",
    "src/pybind/udp_client.cpp",
    # USB
    "src/input/inivation.cpp",
    "src/pybind/usb.cpp",
]

cpp_headers = [
    "src/aedat.hpp",
    # Inputs
    "src/input/inivation.hpp",
    # Python class headers
    "src/pybind/tensor_buffer.hpp",
    "src/pybind/udp_client.hpp",
]

# Define extension based on CUDA availability
cuda_home = cpp_extension._find_cuda_home()
if cuda_home is not None:
    cpp_sources += [
        "src/pybind/tensor_buffer_cuda.cpp",
        "src/pybind/tensor_buffer_kernel.cu",
    ]
    extension = cpp_extension.CUDAExtension(
        name="aestream",
        headers=cpp_headers,
        sources=cpp_sources,
        extra_compile_args={
            "cxx": [
                "-O3",
                "-g",
                "-D_GLIBCXX_USE_CXX11_ABI=0",
                "-fcoroutines",
                "-std=c++20",
                "-I/usr/include/opencv4",
            ],
            "nvcc": ["-O3"],
        },
        libraries=["caer"],
    )
else:
    cpp_sources += [
        "src/pybind/tensor_buffer.cpp",
    ]
    extension = cpp_extension.CppExtension(
        name="aestream",
        headers=cpp_headers,
        sources=cpp_sources,
        extra_compile_args=[
            "-O3",
            "-g",
            "-D_GLIBCXX_USE_CXX11_ABI=0",
            "-fcoroutines",
            "-std=c++20",
            "-I/usr/include/opencv4",
        ],
        libraries=["caer"],
    )

setup(
    name="aestream",
    version="0.3.0",
    author="Jens E. Pedersen, Christian Pehle",
    author_email="jens@jepedersen.dk, christian.pehle@gmail.com",
    url="https://github.com/norse/aestream",
    description="Streaming library for Address-Event Representation (AER) data",
    license="MIT",
    long_description=readme_text,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    install_requires=install_requires,
    setup_requires=["setuptools", "wheel", "torch", "numpy", "ninja"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Software Development :: Libraries",
        "Topic :: System :: Hardware :: Universal Serial Bus (USB)",
    ],
<<<<<<< HEAD
    ext_modules=[extension],
=======
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="aestream",
            headers=cpp_headers,
            sources=cpp_sources,
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-g",
                    "-D_GLIBCXX_USE_CXX11_ABI=0",
                    "-fcoroutines",
                    "-std=c++20",
                    "-I/usr/include/opencv4",
                ],
                'nvcc': ['-O3']
            },
            libraries=["caer"],
        ),
    ],
>>>>>>> 9a2f0b0 (Added CUDA code for tensor buffer)
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
