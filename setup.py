from os import path
from setuptools import setup
from torch.utils import cpp_extension

pwd = path.abspath(path.dirname(__file__))
with open(path.join(pwd, "README.md"), encoding="utf-8") as fp:
    readme_text = fp.read()

with open(path.join(pwd, "requirements.txt")) as fp:
    install_requires = fp.read()

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
    install_requires=install_requires,
    setup_requires=["setuptools", "wheel", "torch"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Software Development :: Libraries",
        "Topic :: System :: Hardware :: Universal Serial Bus (USB)"
    ],
    ext_modules=[
        cpp_extension.CppExtension(
            name="aestream",
            sources=["src/pybind/udp_client.cpp", "src/pybind/udp_listener.cpp"],
            extra_compile_args=["-O3", "-g", "-D_GLIBCXX_USE_CXX11_ABI=0"],
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
