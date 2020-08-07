# aedat - address event encoding library

aedat contains unofficial decoders for the AEDAT 3.1 and 4.0 formats used by 
the dynamic vision sensors of iniVation. In addition it provides support 
for converting polarity events into pytorch sparse tensors, thereby providing
a building block for using dynamic vision sensors in conjunction with pytorch
based machine learning algorithms.

## Dataset viewer

The viewer requires SDL2, [LZ4](https://lz4.github.io/lz4/), and [flatbuffers v. >= 1.12](https://google.github.io/flatbuffers/). [libtorch](https://pytorch.org/cppdocs/installing.html) is needed to build the converter.

To build the viewer and converter binaries
```
export CMAKE_PREFIX_PATH=`absolute path to libtorch/`
mkdir build/
cd build/
cmake -GNinja ..
ninja
```

The viewer can then be used to view the example data
```
./viewer ../example_data/ibm/user01_natural.aedat
```
or to view data from the gesture dataset
```
./viewer ../example_data/ibm/user01_natural.aedat ../example_data/ibm/user01_natural_labels.csv
```

## Python bindings

In order to build and install the python bindings run
```
python setup.py install
```
this assumes that you have pytorch 1.5.1 installed.

A minimal example of using the AEDAT3.1 import functionality is then
```python
import torch # Needs to be first otherwise you will encounter an error
import aedat

data = aedat.AEDAT("example_data/ibm/user01_natural.aedat")
events = aedat.convert_polarity_events(data.polarity_events)
```

An example working with the gesture dataset is
```python
import torch
import aedat

dvs = aedat.DVSGestureData(
    "example_data/ibm/user01_natural.aedat",
    "example_data/ibm/user01_natural_labels.csv"
)

for element in dvs.datapoints
    label = element.label
    events = aedat.convert_polarity_events(element.events)
```

To use the AEDAT4 formatted data you can try the following:

```python
import torch
import aedat
import numpy as np
import mathplotlib.pyplot as plt

data = aedat.AEDAT4("example_data/kth/example.aedat4")

# display the first frame
pixels = data.frames[0].pixels
width, height = data.frames[0].width, data.frames[0].height
im =  np.array(pixels).reshape(height, width, 3)
plt.imshow(im)
plt.show()

# convert the polarity events to a sparse pytorch tensor
events = aedat.convert_polarity_events(data.polarity_events)
```