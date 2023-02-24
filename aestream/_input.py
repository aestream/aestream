from aestream import aestream_ext as ext

try:
    import torch
    USE_TORCH = True
except ImportError:
    USE_TORCH = False

class FileInput(ext.FileInput):
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