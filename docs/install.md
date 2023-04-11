# Install

AEStream is usable both as a command-line binary or Python tool.

| **Source** | **Installation** |
| -------------------- | --- |
| [pip](https://pypi.org/) | <code>pip install aestream <br/> pip install aestream[torch]</code> ([PyTorch support](https://pytorch.com)) |
| [nix](https://nixos.org/) | <code>nix run github:aestream/aestream</code> (CLI) <br/> <code>nix develop github:aestream/aestream</code> (Python environment) |
| [docker](https://docker.com/) | See [Installation documentation](https://aestream.github.io/aestream/install.html) |

Contributions to support AEStream on additional platforms are always welcome.

## Requirements
AEStream relies on new coroutine features and, in turn, compilers that support them such as [GCC 10](https://gcc.gnu.org/).

## Docker install
A `Dockerfile` is available in the root directory and can be built as follows:
```bash
docker build -t aestream .
```

## Source install
Source installations can either be done via pip, which wraps the C++ code, or CMake, where you have to manually configure the project.

### Python source install
```bash
git clone https://github.com/aestream/aestream
cd aestream
pip install .
```

### CMake source install
Note that this will only install the CLI version of AEStream. To have access to the Python features, you must enable the `USE_PYTHON` config.
```bash
git clone https://github.com/aestream/aestream
cd aestream
mkdir build
cd build
cmake -GNinja ..
ninja install
```