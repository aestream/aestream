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
        return getattr(ext.Backend, backend)
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
    def load(self):
        buffer = self.load_all()
        return np.frombuffer(buffer.data, NUMPY_EVENT_DTYPE)

    def read(self, backend: ext.Backend = ext.Backend.Numpy):
        return _read_backend(self, backend, None)


class UDPInput(ext.UDPInput):
    def read(self, backend: ext.Backend = ext.Backend.Numpy):
        return _read_backend(self, backend, None)


try:

    class USBInput(ext.USBInput):
        def read(self, backend: ext.Backend = ext.Backend.Numpy):
            return _read_backend(self, backend, None)

except:
    pass  # Ignore if drivers are not installed

try:

    class SpeckInput(ext.SpeckInput):
        def read(self, backend: ext.Backend = ext.Backend.Numpy):
            return _read_backend(self, backend, None)

except Exception as e:
    pass  # Ignore if Speck/ZMQ isn't installed
