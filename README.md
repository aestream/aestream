# aedat - address event encoding 

This repository contains a decoder and simple viewer for the AEDAT 3.1 format as defined
[here](https://inivation.com/support/software/fileformat/#formats). In addition it provides
support for converting polarity events into a libtorch sparse tensor.

The viewer requires SDL2 and [libtorch](https://pytorch.org/cppdocs/installing.html) is needed to build the converter.

To build the viewer and converter binaries
```
export CMAKE_PREFIX_PATH=`absolute path to libtorch/`
mkdir build/
cd build/
cmake -GNinja ..
```

The viewer can then be used to view the example data
```
./viewer ../example_data/ibm/user01_natural.aedat
```
or to view data from the gesture dataset
```
./viewer ../example_data/ibm/user01_natural.aedat ../example_data/ibm/user01_natural_labels.csv
```

In order to build and install the python bindings run
```
python setup.py install
```
this assumes that you habe pytorch 1.5.1 installed.

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

datapoints = aedat.DVSGestureData("example_data/ibm/user01_natural.aedat", "example_data/ibm/user01_natural_labels.csv")

for element in data.datapoints
    label = element.label
    events = aedat.convert_polarity_events(element.events)
```