# AEDAT - address event encoding and streaming library

AEDAT decodes event-based dynamic-vision system (DVS) data
and streams it to a sink.

We currently support the following inputs:

| Interface | Description |
| --------- | :----------- | 
| DVXplorer       | 640x480 DVS camera, Inivation  |
<!-- | File       | `.aedat` or `.aedat4` | | -->

We currently support the following outputs:

| Interface | Type | Description |
| --------- | :--- | ----------- |
| Ethernet - UDP | Output | Via the [SPIF](https://github.com/SpiNNakerManchester/spif) protocol |
<!-- | File       | Output | `.aedat` or `.aedat4` | -->


## Setup

DVSStream requires [libcaer](https://github.com/inivation/libcaer) and [libtorch](https://pytorch.org/cppdocs/installing.html).

To build the binaries of this repository, run the following code:
```
export CMAKE_PREFIX_PATH=`absolute path to libtorch/`
mkdir build/
cd build/
cmake -GNinja ..
ninja
```
## Acknowledgments

- This library is based on [libcaer](https://github.com/inivation/libcaer) by iniVation AG and interfaced by Philipp Mondorf and Jens E. Pedersen.
- The conversion of polarity events to sparse tensors is based on the [AEDAT](https://github.com/norse/aedat) library by Christian Pehle and Jens. E. Pedersen.