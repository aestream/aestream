from os import path
from setuptools import setup
from torch.utils import cpp_extension

pwd = path.abspath(path.dirname(__file__))
with open(path.join(pwd, "README.md"), encoding="utf-8") as fp:
    readme_text = fp.read()

cpp_sources = [
    # Module code
    "src/pybind/module.cpp",
    "src/pybind/tensor_buffer.cpp",
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

setup(
    name="aestream",
    version="0.1.0",
    author="Jens E. Pedersen, Christian Pehle",
    author_email="jens@jepedersen.dk, christian.pehle@gmail.com",
    url="https://github.com/norse/aestream",
    description="Streaming library for Address-Event Representation (AER) data",
    license="MIT",
    long_description=readme_text,
    python_requires=">=3.6",
    install_requires=["numpy", "torch"],
    setup_requires=["setuptools", "wheel", "torch"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Software Development :: Libraries",
        "Topic :: System :: Hardware :: Universal Serial Bus (USB)",
    ],
    ext_modules=[
        cpp_extension.CppExtension(
            name="aestream",
            headers=cpp_headers,
            sources=cpp_sources,
            include_dirs=["src/", "src/input/", "src/pybind/"],
            extra_compile_args=[
                "-O3",
                "-g",
                "-D_GLIBCXX_USE_CXX11_ABI=0",
                "-fcoroutines",
                "-std=c++20",
            ],
            libraries=["caer"],
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
