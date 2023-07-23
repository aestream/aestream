# AEStream API Prototyping

from typing import Union, Tuple
import numpy as np

try:
    import torch
except:
    "torch not found, proceeding with numpy only"

class FileInput:
    """
    If you see this, its good :)
    """

    def init(): ...
    def read():
        """
        Read :P
        """
        ...

class FileInputReader:
    """
    Load or Iterate Events or Frames from a file

    Supported file formats: aedat, aedat4, dat

    Args:
        file_path (str): Path of file to be used
        resolution (int, int): X,Y resolution of file

    Attributes:
        file_path (str): Path of file to be used
        resolution (int, int): X,Y resolution of file
    """

    def __init__(self, file_path: str, resolution: Tuple[int, int]) -> None: ...
    def load(
        self, frame_ms: int = 0, device: str = "cpu"
    ) -> Union[np.ndarray, torch.tensor]:
        """
        Loads the whole file onto device memory

        Args:
            frame_ms (int): Selects data format. When `0`, events are loaded as is.
                Otherwise, events are accumulated into frames with synchronous period of `frame_ms`
            device (str: `cpu`, `cuda`): Selects the device to which the file is loaded

        Returns:
            Array/Tensor containing the contents of the file
        """
        ...
    def get_iterator(
        self,
        frame_ms: int = 0,
        device: str = "cpu",
        mode: str = "safe",
        memory_limit: int = 1024,
    ) -> FileInputIterator:
        """
        Provides functionality that allows files larger than device memory to be iteratively loaded

        Args:
            frame_ms (int): Selects data format. When `0`, events are loaded as is. Otherwise, events are accumulated into frames with synchronous period of `frame_ms`
            device (str: `cpu`, `cuda`): Selects the device to which the file is loaded.
            mode (str: `safe`, `unsafe`): Selects if slower deterministic (`safe`) or faster nondeterministic (`unsafe`) method is used to create frames. When using `unsafe` the exact frame a event is part of is subject to jitter at runtime.

        Returns:
            A generator that loads events or frames to `device` as specified.
        """
        ...

class FileInputIterator:
    """
    Load or Iterate Events or Frames from a file

    Supported file formats: aedat, aedat4, dat

    Args:
        file_path (str): Path of file to be used
        resolution (int, int): X,Y resolution of file

    Attributes:
        file_path (str): Path of file to be used
        resolution (int, int): X,Y resolution of file
    """

    def __init__(self, file_path: str, resolution: Tuple[int, int]) -> None: ...
