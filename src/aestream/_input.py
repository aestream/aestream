from typing import Any, Optional, Union

from aestream import aestream_ext as ext

# Set Numpy Event dtype
try:
    import numpy as np

    event_dtype = np.dtype(("=V", 13))
    NUMPY_EVENT_DTYPE = np.dtype(
        (
            event_dtype,
            {
                "timestamp": (np.int64, 0),
                "x": (np.int16, 8),
                "y": (np.int16, 10),
                "polarity": (np.byte, 12),
            },
        )
    )

except ImportError as e:
    raise ImportError("Numpy is required but could not be imported", e)


def _convert_parameter_to_backend(backend: Union[ext.Backend, str]):
    if isinstance(backend, ext.Backend):
        return backend
    elif isinstance(backend, str):
        return getattr(ext.Backend, backend.title())
    else:
        raise TypeError("backend must be either ext.Backend or str")


def _read_backend(obj: Any, backend: ext.Backend, population: Optional[Any]):
    backend = _convert_parameter_to_backend(backend)
    if backend == ext.Backend.GeNN:
        obj.read_genn(population.extra_global_params["input"].view)
        population.push_extra_global_param_to_device("input")
        return population.extra_global_params["input"].view
    elif backend == ext.Backend.Jax:
        t = obj.read_buffer()
        return t.to_jax()
    elif backend == ext.Backend.Torch:
        t = obj.read_buffer()
        return t.to_torch()
    else:
        t = obj.read_buffer()
        return t.to_numpy()


class FileInput(ext.FileInput):
    """
    Reads events from a file.

    Parameters:
        filename (str): Path to file.
        shape (tuple): Shape of the camera surface in pixels (X, Y).
        device (str): Device name. Defaults to "cpu"
        ignore_time (bool): Whether to ignore the timestamps for the events when
            streaming. If set to True, the events will be streamed as fast as possible.
            Defaults to False.
    """

    def load(self):
        buffer = self.load_all()
        return np.frombuffer(buffer.data, NUMPY_EVENT_DTYPE)

    def read(self, backend: ext.Backend = ext.Backend.Numpy):
        return _read_backend(self, backend, None)


class UDPInput(ext.UDPInput):
    """
    Reads events from a UDP socket.

    Parameters:
        shape (tuple): Shape of the camera surface in pixels (X, Y).
        device (str): Device name. Defaults to "cpu"
        port (int): Port to listen on. Defaults to 3333.
    """

    def read(self, backend: ext.Backend = ext.Backend.Numpy):
        return _read_backend(self, backend, None)

if "caer" in ext.drivers or "metavision" in ext.drivers:
    class USBInput(ext.USBInput):
        """
        Reads events from a USB camera.

        Parameters:
            shape (tuple): Shape of the camera surface in pixels (X, Y).
            device (str): Device name. Defaults to "cpu"
            device_id (int): Device ID. Defaults to 0.
            device_address (int): Device address, typically on the bus. Defaults to 0.
        """

        def read(self, backend: ext.Backend = ext.Backend.Numpy):
            return _read_backend(self, backend, None)

if "zmq" in ext.drivers:

    class SpeckInput(ext.SpeckInput):
        def read(self, backend: ext.Backend = ext.Backend.Numpy):
            return _read_backend(self, backend, None)
