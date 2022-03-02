# AEStream - Address Event streaming library

AEDAT parses event-based dynamic-vision system (DVS) data
from an input source and streams it to a sink.

AEStream is built in C++, but can be interfaced via CLI or Python (work in progress).

## Usage (Python)

First, install [PyTorch](https://pytorch.org/) and [libcaer](https://github.com/inivation/libcaer). 
Then install `aestream` via pip: `pip install aestream`

```python
# Stream events from a DVS camera over USB at address 2:4
with DVSInput((640, 480)) as stream:
    while True:
        frame = stream.read() # Provides a (640, 480) tensor
        ...
```

```python
# Stream events UDP port 3333
with UDPInput((640, 480)) as stream:
    while True:
        frame = stream.read() # Provides a (640, 480) tensor
        ...
```

More examples can be found in [our example folder](https://github.com/norse/aestream/tree/master/example).

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
| DAVIS           | 346x260 DVS camera, Inivation  | `input inivation X Y davis` |
| DVXplorer       | 640x480 DVS camera, Inivation  | `input inivation X Y dvx` |
| Prophesee       | 640x480 DVS camera, Prophesee  | `input prophesee X` |
| Prophesee       | 1280x720 DVS camera, Prophesee  | `input prophesee X` |
| File            | `.aedat` or `.aedat4` | `input file x.aedat4` |

We currently support the following outputs:

| Output | Description | Usage |
| --------- | ----------- | ----- |
| STDOUT    | Standard output (default output) | `output stdout`
| Ethernet over UDP | Outputs to a given IP and port using the [SPIF protocol](https://github.com/SpiNNakerManchester/spif)  | `output spif 10.0.0.1 1234` |
| File       | Output to file | `output file my_file.txt` |

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

## Acknowledgments

AEStream is created by

* [Jens E. Pedersen](https://www.kth.se/profile/jeped) (@GitHub [jegp](https://github.com/jegp/)), doctoral student at KTH Royal Institute of Technology, Sweden.
* [Christian Pehle](https://www.kip.uni-heidelberg.de/people/10110) (@GitHub [cpehle](https://github.com/cpehle/)), PostDoc at University of Heidelberg, Germany.

The work has received funding from the EC Horizon 2020 Framework Programme under Grant Agreements 785907 and 945539 (HBP) and by the Deutsche Forschungsgemeinschaft (DFG, German Research Fundation) under Germany's Excellence Strategy EXC 2181/1 - 390900948 (the Heidelberg STRUCTURES Excellence Cluster).

Thanks to [Philipp Mondorf](https://github.com/PMMon) for interfacing with Metavision SDK and preliminary network code.
