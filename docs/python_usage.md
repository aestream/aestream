# Python usage

AEStream provides three Python classes that can be used to stream events: `FileInput`, `USBInput`, and `UDPInput`. The `FileInput` class also supports deterministic reading, which is useful to load and process files and datasets.

For each input below we provide example scripts. 
More examples can be found in [our example folder](https://github.com/aestream/aestream/tree/master/example).

Please note the examples may require additional dependencies (such as [Norse](https://github.com/norse/norse) for spiking networks or [PySDL](https://github.com/py-sdl/py-sdl2) for rendering). To install all the requirements, simply stand in the `aestream` root directory and run `pip install -r example/requirements.txt`

## `FileInput`

AEStream can process fixed input sources like files like so:

```python
FileInput("file", (640, 480)).load()
```

> Example: [Reading a file](https://github.com/aestream/aestream/blob/main/example/file_read.py): `python3 example/file_read.py`

> Example: [Streaming a file](https://github.com/aestream/aestream/blob/main/example/file_stream.py): `python3 example/file_stream.py`

## `USBInput`

AEStream also supports streaming data in real-time *without strict guarantees on orders*. 
This is particularly useful in real-time scenarios, for instance when operating with `USBInput` or `UDPInput`

```python
# Stream events from a DVS camera over USB
with USBInput((640, 480)) as stream:
    while True:
        frame = stream.read() # Provides a (640, 480) tensor
        ...
```

> Example: [Displaying a video of streaming events](https://github.com/aestream/aestream/blob/main/example/usb_video.py): `python3 example/usb_video.py`

> Example: [Detecting edges with a neural network](https://github.com/aestream/aestream/blob/main/example/usb_edgedetection.py): `python3 example/usb_edgedetection.py`

![](../example/usb_edgedetection.gif)

## `UDPInput`

AEStream implements the ["SPIF" UDP event protocol](https://github.com/SpiNNakerManchester/spif/tree/master/spiffer) that allows us to both send, but also receive events. The `UDPInput` reads events and serves them ready for processing in frames as below. Note that the events are expected to follow the protocol.

```python
# Stream events from UDP port 3333 (default)
with UDPInput((640, 480), port=3333) as stream:
    while True:
        frame = stream.read() # Provides a (640, 480) tensor
        ...
```

> Example: [Print number of events received over UDP](https://github.com/aestream/aestream/blob/main/example/udp_client.py): `python3 example/udp_client.py`

> Example: [Record frames over UDP](https://github.com/aestream/aestream/blob/main/example/udp_video.py): `python3 example/udp_video.py`