from aestream import aestream_ext as ext

# Set Numpy Event dtype
try:
    import numpy as np
    event_dtype = np.dtype(("=V", 13))
    NUMPY_EVENT_DTYPE = np.dtype((event_dtype, {"timestamp": (np.int64, 0), "x": (np.int16, 8), "y": (np.int16, 10), "polarity": (np.byte, 12)}))

except ImportError as e:
    raise ImportError("Numpy is required but could not be imported", e)

try:
    import torch
    USE_TORCH = True
except ImportError:
    USE_TORCH = False

class FileInput(ext.FileInput):

    def load(self):
        buffer = self.load_all()
        return np.frombuffer(buffer.data, NUMPY_EVENT_DTYPE)

    def read(self):
        t = self.read_buffer()
        if USE_TORCH:
            return t.to_torch()
        else:
            return t.to_numpy()

class UDPInput(ext.UDPInput):
    def read(self):
        t = self.read_buffer()
        if USE_TORCH:
            return t.to_torch()
        else:
            return t.to_numpy()