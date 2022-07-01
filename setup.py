from os import path
from setuptools import setup
from torch.utils import cpp_extension

pwd = path.abspath(path.dirname(__file__))
with open(path.join(pwd, "README.md"), encoding="utf-8") as fp:
    readme_text = fp.read()

setup(
    name="aestream",
    version="0.0.1",
    author="Jens E. Pedersen, Christian Pehle",
    author_email="jens@jepedersen.dk, christian.pehle@gmail.com",
    description="Streaming library for Address-Event Representation (AER) data",
    long_description=readme_text,
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.6",
    install_requires=["numpy", "torch>=1.9.0"],
    setup_requires=["setuptools", "torch"],
    ext_modules=[
        cpp_extension.CppExtension(
            name="aestream",
            sources=["src/pybind/udp_client.cpp", "src/pybind/udp_listener.cpp"],
            extra_compile_args=["-O3", "-g", "-D_GLIBCXX_USE_CXX11_ABI=0"],
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
