# Command-line interface

The AEStream command-line interface (CLI) provides a mean to **read** some input and **stream** to some output.
We support a wide range of inputs and outputs.

The CLI interface *requires* an input, but an *optional* output and takes the following form:
```bash
aestream input <input source> [output <output sink>]
```

## Supported Inputs and Outputs

| Input | Description | Usage |
| --------- | :----------- | ----- |
| DAVIS, DVXPlorer | [Inivation](https://inivation.com/) DVS Camera over USB | `input inivation` |
| File             | [AEDAT file format](https://gitlab.com/inivation/inivation-docs/blob/master/Software%20user%20guides/AEDAT_file_formats.md) as `.aedat`, `.aedat4`, or `.dat` | `input file x.aedat4` |

<!-- | EVK Cameras      | [Prophesee](https://www.prophesee.ai/) DVS camera over USB  | `input prophesee` | -->

| Output | Description | Usage |
| --------- | ----------- | ----- |
| STDOUT    | Standard output (default output) | `output stdout`
| Ethernet over UDP | Outputs to a given IP and port using the [SPIF protocol](https://github.com/SpiNNakerManchester/spif)  | `output udp 10.0.0.1 1234` |
| `.aedat4` file  | Output to [`.aedat4` format](https://gitlab.com/inivation/inivation-docs/blob/master/Software%20user%20guides/AEDAT_file_formats.md#aedat-40) | `output file my_file.aedat4` |
| CSV file       | Output to comma-separated-value (CSV) file format | `output file my_file.txt` |