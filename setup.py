import os
from skbuild import setup

pwd = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(pwd, "README.md"), encoding="utf-8") as fp:
    readme_text = fp.read()

# C++ config
cmake_args = [
    "-DWITH_PYTHON=1",
    "-D_GLIBCXX_USE_CXX11_ABI=0"
    # "-DCUDA_PATH=/usr/local/cuda",
    # "-DCMAKE_CUDA_FLAGS=-arch=sm_70",
    # "-DCMAKE_CUDA_ARCHITECTURES=50;52;60;61;70;75;80;86",
    #"-DCMAKE_CXX_FLAGS=-fno-lto"
    
    # f"-DSKBUILD_MODULE_PATH={python_path}"
    # f"-DCMAKE_PREFIX_PATH='{os.path.dirname(torch.__file__)};{torch.utils.cmake_prefix_path}'",
    # f"-DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI={1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0} -fPIC'",
]

# Setuptools entrypoint
setup(
    name="aestream",
    version="0.5.0",
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
    extras_require={"torch": ["torch"]},
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Software Development :: Libraries",
        "Topic :: System :: Hardware :: Universal Serial Bus (USB)",
    ],
    cmake_args=cmake_args,
)
