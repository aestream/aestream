# AEStream - Address Event streaming library

<p align="center">
    <a href="https://github.com/norse/aestream/actions">
        <img src="https://github.com/norse/aestream/workflows/Build%20and%20test/badge.svg" alt="Test status"></a>
    <a href="https://pypi.org/project/aestream/" alt="PyPi">
        <img src="https://img.shields.io/pypi/v/aestream" />
    </a>
    <a href="https://github.com/norse/aestream/pulse" alt="Activity">
        <img src="https://img.shields.io/github/last-commit/norse/aestream" />
    </a>
    <a href="https://discord.gg/7fGN359">
        <img src="https://img.shields.io/discord/723215296399147089"
            alt="chat on Discord"></a>
    <a href="https://www.codacy.com/gh/norse/aestream/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=norse/aestream&amp;utm_campaign=Badge_Grade"><img src="https://app.codacy.com/project/badge/Grade/0a04a852daf540a9b9bbe9d78df9eea7"/></a>
    <a href="https://doi.org/10.5281/zenodo.6322829"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6322829.svg" alt="DOI"></a>
</p>


AEDAT parses event-based dynamic-vision system (DVS) data
from an input source and streams it to a sink.

AEStream is built in C++, but can be interfaced via CLI or Python (work in progress).

## Usage (Python)

First, install [PyTorch](https://pytorch.org/) and [libcaer](https://gitlab.com/inivation/dv/libcaer/). 
Then install `aestream` via pip: `pip install aestream`

```python
# Stream events from a DVS camera over USB
with DVSInput((640, 480)) as stream:
    while True:
        frame = stream.read() # Provides a (640, 480) tensor
        ...
```

```python
# Stream events from UDP port 3333 (default)
with UDPInput((640, 480), port=3333) as stream:
    while True:
        frame = stream.read() # Provides a (640, 480) tensor
        ...
```

More examples can be found in [our example folder](https://github.com/norse/aestream/tree/master/example).
Please note the examples may require additional dependencies (such as [Norse](https://github.com/norse/norse) for spiking networks or [PySDL](https://github.com/py-sdl/py-sdl2) for rendering). To install all the requirements, simply stand in the `aestream` root directory and run `pip install -r example/requirements.txt`

## Usage (CLI)

AEStream produces a binary `stream` that requires you to specify an `input` source and an optional `output` source (defaulting to STDOUT).
The general syntax is as follows (input is required, output is optional):

```bash
aestream input <input source> [output <output sink>]
```
## Supported Inputs and Outputs

We currently support the following inputs:

| Input | Description | Usage |
| --------- | :----------- | ----- |
| DAVIS           | 346x260 DVS camera with USB address `X:Y`, Inivation  | `input inivation X Y davis` |
| DVXplorer       | 640x480 DVS camera with USB address `X:Y`, Inivation  | `input inivation X Y dvx` |
| Prophesee       | 640x480 DVS camera with USB address `X`, Prophesee  | `input prophesee X` |
| Prophesee       | 1280x720 DVS camera with USB address `X`, Prophesee  | `input prophesee X` |
| File            | `.aedat` or `.aedat4` | `input file x.aedat4` |

We currently support the following outputs:

| Output | Description | Usage |
| --------- | ----------- | ----- |
| STDOUT    | Standard output (default output) | `output stdout`
| Ethernet over UDP | Outputs to a given IP and port using the [SPIF protocol](https://github.com/SpiNNakerManchester/spif)  | `output udp 10.0.0.1 1234` |
| `.aedat4` file  | Output to [`.aedat4` format](https://gitlab.com/inivation/inivation-docs/blob/master/Software%20user%20guides/AEDAT_file_formats.md#aedat-40) | `output file my_file.aedat4` |
| CSV file       | Output to comma-separated-value (CSV) file format | `output file my_file.txt` |

### CLI examples

| Example | Syntax |
| ------------- | ------------------------------|
| Read file to STDOUT | `aestream input file example/davis.aedat4` |
| Stream DVS Davis346 (USB 0:2) by iniVation AG to STDOUT (Note, requires Inivation libraries) | `aestream input inivation 0 2 davis output stdout` |
| Stream Prophesee 640x480 (serial Prophesee:hal_plugin_gen31_fx3:00001464) to STDOUT (Note, requires Metavision SDK) | `aestream input prophesee Prophesee:hal_plugin_gen31_fx3:00001464 output stdout` |
| Read file to remote machine X.X.X.X | `aestream input file example/davis.aedat4 output udp X.X.X.X` |

## Setup (C++)

AEStream requires [libtorch](https://pytorch.org/cppdocs/installing.html). [Metavision SDK](https://docs.prophesee.ai/stable/metavision_sdk/index.html), [libcaer](https://github.com/inivation/libcaer) and [OpenCV](https://github.com/opencv/opencv) are optional dependencies, but are needed for some functionality.

AEStream is based on [C++20](https://en.cppreference.com/w/cpp/20). Since C++20 is not yet fully supported by all compilers, we recommend using `GCC >= 10.2`. 

To build the binaries of this repository, run the following code:
```
export CMAKE_PREFIX_PATH=`absolute path to libtorch/`
# Optional: Ensure paths to libcaer, Metavision, or OpenCV is in place
mkdir build/
cd build/
cmake -GNinja ..
ninja
```

If your default C++ compiler doesn't support C++ 20, you will have to install an up-to-date compiler and provide the environmental variable `CXX`.
For instance like this: `CXX=/path/to/g++ cmake -GNinja ..`

### Inivation cameras
For [Inivation](https://inivation.com/) cameras, the [libcaer](https://gitlab.com/inivation/dv/libcaer/) library needs to be available, either by a `-DCMAKE_PREFIX_PATH` flag to `cmake` or included in the `PATH` environmental variable.
For examble: `cmake -GNinja -DCMAKE_PREFIX_PATH=/path/to/libcaer`.
Inivation made the library available for most operating systems, but you may have to build it yourself.

### Prophesee cameras
For [Prophesee](https://www.prophesee.ai/) cameras, a version of the [Metavision SDK](https://www.prophesee.ai/metavision-intelligence/) needs to be present.
The open-source version the SDK `openeb` is available with installation instructions at https://github.com/prophesee-ai/openeb.
Using `openeb`, it should be sufficient to install it using `cmake && make && make install` to put it in your path.
Otherwise, you can point to it using the `-DCMAKE_PREFIX_PATH` option in `cmake`.

## Acknowledgments

AEStream is created by

* [Jens E. Pedersen](https://www.kth.se/profile/jeped) (@GitHub [jegp](https://github.com/jegp/)), doctoral student at KTH Royal Institute of Technology, Sweden.
* [Christian Pehle](https://www.kip.uni-heidelberg.de/people/10110) (@GitHub [cpehle](https://github.com/cpehle/)), PostDoc at University of Heidelberg, Germany.

The work has received funding from the EC Horizon 2020 Framework Programme under Grant Agreements 785907 and 945539 (HBP) and by the Deutsche Forschungsgemeinschaft (DFG, German Research Fundation) under Germany's Excellence Strategy EXC 2181/1 - 390900948 (the Heidelberg STRUCTURES Excellence Cluster).

Thanks to [Philipp Mondorf](https://github.com/PMMon) for interfacing with Metavision SDK and preliminary network code.

## Citation

Please cite `aestream` if you use it in your work:

```bibtex
@software{aestream2022,
  author       = {Pedersen, Jens Egholm and
                  Pehle, Christian-Gernot},
  title        = {AEStream - Address Event Streaming library},
  month        = {August},
  year         = 2022,
  publisher    = {Zenodo},
  version      = {0.3.0},
  doi          = {10.5281/zenodo.6322829},
  url          = {https://doi.org/10.5281/zenodo.6322829}
}
```
