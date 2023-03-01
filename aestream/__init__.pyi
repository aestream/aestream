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


class FileInputIterator:
    """
    Iterate Events or Frames from a file using a generator
    Supported file formats: aedat, aedat4, dat

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
        - `memory_limit: int = 1024` Maximum device memory to be uses in MB.

    ### Attributes:
        - `file_path: str` Path of file to be used.
        - `resolution: (int, int)` X,Y resolution of file.
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

    def __next__():
        ...
