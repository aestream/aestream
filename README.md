# AEStream - Address Event encoding and streaming library

AEStream parses event-based dynamic-vision system (DVS) data from an input source and streams it to a sink.

## Usage

AEStream produces a binary `stream` that requires you to specify an `input` source and an optional `output` source (defaulting to STDOUT).

Here are a few examples of input/output combinations.

### File Input

```bash
# Read file to STDOUT
stream input file example/davis.aedat4 
```

### Inivation Camera Input

```bash
# Stream DVS Davis346 (USB 0:2) by iniVation AG to STDOUT
stream input inivation 0 2 davis output stdout
```

### Prophesee Camera Input

```bash
# Stream Prophesee 640x480 (serial Prophesee:hal_plugin_gen31_fx3:00001464) to STDOUT
stream input prophesee Prophesee:hal_plugin_gen31_fx3:00001464 output stdout
```


## Supported Inputs and Outputs

We currently support the following inputs:

| Interface | Description | Usage |
| --------- | :----------- | ----- |
| DAVIS           | 346x260 DVS camera, Inivation  | `input inivation X X davis` |
| DVXplorer       | 640x480 DVS camera, Inivation  | `input inivation X X dvx` |
| Prophesee       | 640x480 DVS camera, Prophesee  | `input prophesee X` |
| Prophesee       | 1280x720 DVS camera, Prophesee  | `input prophesee X` |
| File            | `.aedat` or `.aedat4` | `input file x.aedat4` |

We currently support the following outputs:

| Interface | Description | Usage |
| --------- | ----------- | ----- |
| STDOUT    | Standard output | (empty) or `output stdout` 
<!-- | Ethernet - UDP | Output | Via the [SPIF](https://github.com/SpiNNakerManchester/spif) protocol | -->
<!-- | File       | Output | `.aedat` or `.aedat4` | -->


## Setup

AEStream requires [libcaer](https://github.com/inivation/libcaer), [libtorch](https://pytorch.org/cppdocs/installing.html), [Metavision SDK](https://docs.prophesee.ai/stable/metavision_sdk/index.html) and [OpenCV](https://github.com/opencv/opencv).

As AEDATStream uses coroutines, it is based on [C++20](https://en.cppreference.com/w/cpp/20). Since C++20 is not yet fully supported by all compilers, we recommend using `GCC >= 10.2`. 


To build the binaries of this repository, run the following code:
```
export CMAKE_PREFIX_PATH=`absolute path to libtorch/`
mkdir build/
cd build/
cmake -GNinja ..
ninja
```
## Acknowledgments

- The conversion of polarity events to sparse tensors is based on the [AEDAT](https://github.com/norse/aedat) library by Christian Pehle and Jens. E. Pedersen.
- This library is based on [libcaer](https://github.com/inivation/libcaer) by iniVation AG and interfaced by Philipp Mondorf and Jens E. Pedersen.