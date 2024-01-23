# Installation

AEStream is usable both as a command-line binary or Python tool.

| **Source** | **Installation**
| -------------------- | --- |
| [pip](https://pypi.org/) | <code>pip install aestream |
| [nix](https://nixos.org/) | <code>nix run github:aestream/aestream</code> (CLI) <br/> <code>nix develop github:aestream/aestream</code> (Python environment) |
| [docker](https://docker.com/) | See [Installation documentation](https://aestream.github.io/aestream/install.html) |

Contributions to support AEStream on additional platforms are always welcome.

## Requirements
AEStream relies on new coroutine features and, in turn, compilers that support them such as [GCC 10](https://gcc.gnu.org/).

If you are on older machines that default to GCC versions *below* 10, you can enforce a specific version by setting the environmental variable `CXX`. Here is an example for pip:

`CXX=/path/to/g++-10 pip install aestream`

---

## Installing via pip
Run `pip install aestream`

Installing via pip is the most convenient method to install AEStream and provides access to both the [command-line interface](cli) and the [Python API](python_usage).
Pip is the [Python package manager](https://pip.pypa.io/en/stable/installation/) that is accessible on most computers.

### Event camera drivers
AEStream can read from [Inivation](https://gitlab.com/inivation/dv/libcaer/) or [Prophesee](https://github.com/prophesee-ai/openeb/) event cameras, *given that the drivers are installed*.
That is, the **drivers must be installed before aestream**.

1. Follow the instructions at https://gitlab.com/inivation/dv/libcaer/ to install the Inivation drivers and/or https://github.com/prophesee-ai/openeb/ to install the Prophesee drivers
2. Install AEStream with pip: `pip install aestream --no-binary -v`
    * The `--no-binary` flag forces a recompilation, which will detect the event camera drivers, if 
    * The `-v` flag enables verbose output, which will show you if the drivers were detected
    * If you already have an installation, you need to re-install
3. Ensure that the drivers were detected by inspecting the installation log
    * If successful, you should see messages like `-- Inivation dependencies (libcaer) found at /usr/local/lib/cmake/libcaer`
4. Test your installation by running the [`usb_video.py` example](https://github.com/aestream/aestream/blob/main/example/usb_video.py). If a window with streaming events pops up, you are successful!

Once installed, the cameras can be used in both [CLI](cli) or [Python](python_usage).
We are working to automate the driver installation as a part of AEStream. Help is welcome and appreciated!

## Installing via Docker
A `Dockerfile` is available in the root directory and can be built with the `Dockerfile` available in the aestream root directory as follows:
```bash
docker build -t aestream .
```

## Installing via source
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

If your default C++ compiler doesn't support C++ 20, you will have to install an up-to-date compiler and provide the environmental variable `CXX`.
For instance like this: `CXX=/path/to/g++-10 cmake -GNinja ..`