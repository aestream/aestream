
<img src="https://github.com/aestream/aestream/raw/main/logo.png" />

<p align="center">
    <a href="https://github.com/aestream/aestream/actions">
        <img src="https://github.com/aestream/aestream/workflows/Build%20and%20test/badge.svg" alt="Test status"></a>
    <a href="https://pypi.org/project/aestream/" alt="PyPi">
        <img src="https://img.shields.io/pypi/v/aestream" />
    </a>
    <a href="https://github.com/aestream/aestream/pulse" alt="Activity">
        <img src="https://img.shields.io/github/last-commit/aestream/aestream" />
    </a>
    <a href="https://discord.gg/7fGN359">
        <img src="https://img.shields.io/discord/723215296399147089"
            alt="chat on Discord"></a>
    <a href="https://www.codacy.com/gh/aestream/aestream/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=aestream/aestream&amp;utm_campaign=Badge_Grade"><img src="https://app.codacy.com/project/badge/Grade/0a04a852daf540a9b9bbe9d78df9eea7"/></a>
    <a href="https://doi.org/10.5281/zenodo.6322829"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6322829.svg" alt="DOI"></a>
</p>

AEStream effiently sends event-based data from A to B.
AEStream can be used from the command-line, via Python, or as a C++ library.
We support multiple inputs and outputs, providing seamless integration with files, [event cameras](https://en.wikipedia.org/wiki/Event_camera), network data, Python libraries via Numpy or PyTorch, and visualization tools.

<img src="https://github.com/aestream/aestream/raw/main/docs/aestream_flow.png" />

Read more about the inner workings of the library in [the AEStream publication](https://jegp.github.io/aestream-paper/).

## Installation

> Read more in our [installation guide](https://aestream.github.io/aestream/install.html)

The fastest way to install AEStream is by using pip: `pip install aestream`.
See below for other sources.

| **Source** | **Installation** | **Description** |
| -------------------- | --- | --- |
| [pip](https://pypi.org/) | <code>pip install aestream</code> <br/> <code>pip install aestream[torch]</code> <br/> <code>pip install aestream --no-binary</code> | <br/> [PyTorch support](https://pytorch.com) <br/> <a href="https://aestream.github.io/aestream/install.html#Event-camera-support">Requires camera drivers*</a> |
| [nix](https://nixos.org/) | <code>nix run github:aestream/aestream</code> <br/> <code>nix develop github:aestream/aestream</code> | Command-line interface <br/> Python environment |
| [docker](https://docker.com/) | See <a href="https://aestream.github.io/aestream/install.html">Installation documentation</a> |

<span style="font-size: 80%">
* Event camera support requires available drivers. <a href="https://aestream.github.io/aestream/install.html">A step-by-step guide is available in our documentation</a>.
</span>

Contributions to support AEStream on additional platforms are always welcome.

## Usage (Python)

> Read more in our [Python usage guide](https://aestream.github.io/aestream/python_usage.html)

AEStream can process fixed input sources like files like so:

```python
FileInput("file", (640, 480)).load()
```

## Usage: stream real-time data in Python
AEStream also supports streaming data in real-time *without strict guarantees on orders*. 
This is particularly useful in real-time scenarios, for instance when operating with `USBInput` or `UDPInput`

```python
# Stream events from a DVS camera over USB
with USBInput((640, 480)) as stream:
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

More examples can be found in [our example folder](https://github.com/aestream/aestream/tree/master/example).
Please note the examples may require additional dependencies (such as [Norse](https://github.com/norse/norse) for spiking networks or [PySDL](https://github.com/py-sdl/py-sdl2) for rendering). To install all the requirements, simply stand in the `aestream` root directory and run `pip install -r example/requirements.txt`

### Example: real-time edge detection with spiking neural networks

![](https://media.githubusercontent.com/media/aestream/aestream/main/example/usb_edgedetection.gif)

We stream events from a camera connected via USB and process them on a GPU in real-time using the [spiking neural network library, Norse](https://github.com/norse/norse) using fewer than 50 lines of Python.
The left panel in the video shows the raw signal, while the middle and right panels show horizontal and vertical edge detection respectively.
The full example can be found in [`example/usb_edgedetection.py`](https://github.com/aestream/aestream/blob/main/example/usb_edgedetection.py)

## Usage (CLI)
> Read more in our [CLI usage documentation page](https://aestream.github.io/aestream/install.html)

Installing AEStream also gives access to the command-line interface (CLI) `aestream`.
To use `aestraem`, simply provide an `input` source and an optional `output` sink (defaulting to STDOUT):

```bash
aestream input <input source> [output <output sink>]
```
## Supported Inputs and Outputs

| Input | Description | Example usage |
| --------- | :----------- | ----- |
| DAVIS, DVXPlorer | [Inivation](https://inivation.com/) DVS Camera over USB | `input inivation` |
| EVK Cameras      | [Prophesee](https://www.prophesee.ai/) DVS camera over USB  | `input prophesee` |
| File             | Reads `.aedat`, `.aedat4`, `.csv`, `.dat`, or `.raw` files | `input file x.aedat4` |
| [SynSense Speck](https://www.synsense.ai/products/speck-2/) | Stream events via ZMQ | `input speck` |
| UDP network | Receives stream of events via the [SPIF protocol](https://github.com/SpiNNakerManchester/spif/tree/master/spiffer) | `input udp`

| Output | Description | Example usage |
| --------- | ----------- | ----- |
| STDOUT    | Standard output (default output) | `output stdout`
| Ethernet over UDP | Outputs to a given IP and port using the [SPIF protocol](https://github.com/SpiNNakerManchester/spif)  | `output udp 10.0.0.1 1234` |
| File: `.aedat4`  | Output to [`.aedat4` format](https://gitlab.com/inivation/inivation-docs/blob/master/Software%20user%20guides/AEDAT_file_formats.md#aedat-40) | `output file my_file.aedat4` |
| File: `.csv`       | Output to comma-separated-value (CSV) file format | `output file my_file.csv` |
| Viewer | View live event stream | `output view`

## CLI examples

| Example | Syntax |
| ------------- | ------------------------------|
| View live stream of Inivation camera (requires Inivation drivers) | `aestream input inivation output view` |
| Stream Prophesee camera over the network to 10.0.0.1 (requires Metavision SDK) | `aestream input output udp 10.0.0.1` |
| Convert `.dat` file to `.aedat4` | `aestream input example/sample.dat output file converted.aedat4` |

## Acknowledgments

AEStream is developed by (in alphabetical order):

* Cameron Barker (@GitHub [cameron-git](https://github.com/cameron-git/))
* [Juan Pablo Romero Bermudez](https://www.kth.se/profile/jprb) (@GitHub [jpromerob](https://github.com/jpromerob/))
* Alexander Hadjivanov (@Github [cantordust](https://github.com/cantordust))
* Emil Jansson (@GitHub [emijan-kth](https://github.com/emijan-kth))
* [Jens E. Pedersen](https://www.kth.se/profile/jeped) (@GitHub [jegp](https://github.com/jegp/))
* [Christian Pehle](https://www.kip.uni-heidelberg.de/people/10110) (@GitHub [cpehle](https://github.com/cpehle/))

The work has received funding from the EC Horizon 2020 Framework Programme under Grant Agreements 785907 and 945539 (HBP) and by the Deutsche Forschungsgemeinschaft (DFG, German Research Fundation) under Germany's Excellence Strategy EXC 2181/1 - 390900948 (the Heidelberg STRUCTURES Excellence Cluster).

Thanks to [Philipp Mondorf](https://github.com/PMMon) for interfacing with Metavision SDK and preliminary network code.

<a href="https://github.com/aestream/aestream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=aestream/aestream" />
</a>


## Citation

Please cite `aestream` if you use it in your work:

```bibtex
@misc{aestream,
  doi = {10.48550/ARXIV.2212.10719},
  url = {https://arxiv.org/abs/2212.10719},
  author = {Pedersen, Jens Egholm and Conradt, Jörg},
  title = {AEStream: Accelerated event-based processing with coroutines},
  publisher = {arXiv},
  year = {2022},
}

```
