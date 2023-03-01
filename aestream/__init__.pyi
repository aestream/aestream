# AEStream API Prototyping

from typing import Union, Tuple
import numpy as np

try:
    import torch
except:
    "torch not found, proceeding with numpy only"


def load_file(
    self,
    frame_ms: int = 0,
    device: str = "cpu",
) -> Union[np.ndarray, torch.tensor]:
    """
    Loads the whole file onto device memory

    Args:
        - `frame_ms: int = 0` Selects data format.
            - `0` Events are loaded as events of dims `[x, y, polarity, time]`
            - `else` Frames generated with synchronous period of `frame_ms`
        - `device: str = "cpu"` Selects the device to which the file is loaded
            - `cpu` CPU
            - `cuda` Nvidia CUDA compatible GPU

    Returns:
        Array/Tensor containing the contents of the file
    """
    ...


class FileInIterator:
    """
    ### File Input Iterator
    Lazily iterate through Events or Frames from a file.
    ??? Explain how the batches work and that all constraints are kept.

    Supported file formats:
    - dat
    - aedat
    - aedat4

    ### Args:
        - `file_path: str` Path of file to be used.
        - `resolution: (int, int)` X,Y resolution of file.
        - `frame_ms: int = 0` Selects data format.
            - `0` Events are loaded as events of dims `[x, y, polarity, time]`
            - `else` Frames generated with synchronous period of `frame_ms`
        - `device: str = "cpu"` Selects the device to which the file is loaded
            - `cpu` CPU
            - `cuda` Nvidia CUDA compatible GPU
        - `mode: str = "safe"` Selects which method is used to create frames.
            - `safe` Slower deterministic method.
            - `unsafe` Faster non-deterministic method. The exact frame a event is part of is subject to jitter at runtime.
        - `n_events: int = 0` Maximum number of events in a batch
            - `0` Disables this limit
        - `n_frames: int = 0` Maximum number of frames in a batch
            - `0` Disables this limit
        - `memory_limit: int = 1024` Maximum device memory to be used in MB.

    ### Methods: ??? needs expanding
    ??? needs adding to other classes
        - `__next__()` Returns next batch ??? needs expanding

    ### Attributes:
        - `file_path: str` Path of file to be used.
        - `resolution: (int, int)` X,Y resolution of file.

    ### Examples
    ```
    with FileInputIterator("path_to_my_file.dat", (640, 480)) as file_iterator:
        for event in file_iterator:
            print(event)
    ```
    """

    def __init__(
        self,
        file_path: str,
        resolution: Tuple[int, int],
        frame_ms: int = 0,
        device: str = "cpu",
        mode: str = "safe",
        n_events: int = 0,
        n_frames: int = 0,
        memory_limit: int = 1024,
    ) -> None: ...

    # ??? do we need to create interface for __next__()
    # We need to discuss how the iterator/generator is produced with C++ and nanobind so its in a compatible form


class NetworkInIterator:
    """
    ### Network Input Iterator
    Lazily iterate through Events or Frames from a network stream.

    Supported network protocols:
    - UDP ??? confirm

    ### Args:
        - `stream_path: str` ??? stream id method here
        - `resolution: (int, int)` X,Y resolution of file.
        - `frame_ms: int = 0` Selects data format. ??? This needs to be included right? Would have to match the send format, unless send is always events.
            - `0` Events are loaded as events of dims `[x, y, polarity, time]`
            - `else` Frames generated with synchronous period of `frame_ms`
        - `device: str = "cpu"` Selects the device to which the file is loaded
            - `cpu` CPU
            - `cuda` Nvidia CUDA compatible GPU
        - ??? another mode: UDP, TCP...
        - `mode: str = "safe"` Selects which method is used to create frames. ??? confirming this is an option, "No (Yes? Blocking?)" in rfc.
            - `safe` Slower deterministic method.
            - `unsafe` Faster non-deterministic method. The exact frame a event is part of is subject to jitter at runtime.
        - `memory_limit: int = 1024` Maximum device memory to be used in MB.

    ### Attributes: ??? update
        - `resolution: (int, int)` X,Y resolution of file.

    ### Examples
    ```
    ```
    """

    def __init__(  # Needs to be updated to match
        self,
        stream_path: str,
        resolution: Tuple[int, int],
        frame_ms: int = 0,
        device: str = "cpu",
        mode: str = "safe",
        memory_limit: int = 1024,
    ) -> None: ...


class NeuroInIterator:
    """
    ### Neuromorphic Input Iterator
    Lazily iterate through Events or Frames from a neuromorphic device.

    Supported neuromorphic devices:
    - Prophesee
        - EKV3
        - EKV4
    - Innovation ??? expand


    ### Args:
        - neurmorphic device path ??? how is this done, also does device type need specified separate.
        - `resolution: (int, int)` X,Y resolution of file. ??? is this always x,y. most obvious would be neuromorphic chip output that preprocesses event camera data. But idk, are there event microphones or other devices which may not use x,y (or more generally exactly 2 non-time data dimensions)
        - `frame_ms: int = 0` Selects data format. ??? same as above, this is specific to cameras, I guess non-visual event data processing is suited for another project
            - `0` Events are loaded as events of dims `[x, y, polarity, time]`
            - `else` Frames generated with synchronous period of `frame_ms`
        - `device: str = "cpu"` Selects the device to which the file is loaded
            - `cpu` CPU
            - `cuda` Nvidia CUDA compatible GPU
        - `mode: str = "safe"` Selects which method is used to create frames.
            - `safe` Slower deterministic method.
            - `unsafe` Faster non-deterministic method. The exact frame a event is part of is subject to jitter at runtime.
        - `memory_limit: int = 1024` Maximum device memory to be used in MB.

    ### Attributes: ??? update
        - `resolution: (int, int)` X,Y resolution of file.

    ### Examples
    ```
    ```
    """

    def __init__(
        self,
        file_path: str,
        resolution: Tuple[int, int],
        frame_ms: int = 0,
        device: str = "cpu",
        mode: str = "safe",
        memory_limit: int = 1024,
    ) -> None: ...
