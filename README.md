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

AEDATStream requires [libcaer](https://github.com/inivation/libcaer), [libtorch](https://pytorch.org/cppdocs/installing.html) and [OpenCV](https://github.com/opencv/opencv).

As it uses coroutines, AEDATStream is based on [C++20](https://en.cppreference.com/w/cpp/20). Since C++20 is not yet fully supported by all compilers, we recommend using `GCC >= 10.2`. 


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