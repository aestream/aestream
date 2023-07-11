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

    def read_genn(self, population):
        # **YUCK** I would like to implement this with a mixin
        # to reduce copy-paste but seemingly nanobind doesn't like this
        # Read from stream into GeNN-owned memory
        super().read_genn(population.extra_global_params["input"].view)

        # Copy data to device
        # **NOTE** this may be a NOP if CPU backend is used
        population.push_extra_global_param_to_device("input")

        return population.extra_global_params["input"].view


class UDPInput(ext.UDPInput):
    def read(self):
        t = self.read_buffer()
        if USE_TORCH:
            return t.to_torch()
        else:
            return t.to_numpy()

    def read_genn(self, population):
        # **YUCK** I would like to implement this with a mixin
        # to reduce copy-paste but seemingly nanobind doesn't like this
        # Read from stream into GeNN-owned memory
        super().read_genn(population.extra_global_params["input"].view)

        # Copy data to device
        # **NOTE** this may be a NOP if CPU backend is used
        population.push_extra_global_param_to_device("input")

        return population.extra_global_params["input"].view


try:

    class USBInput(ext.USBInput):
        def read(self):
            t = self.read_buffer()
            if USE_TORCH:
                return t.to_torch()
            else:
                return t.to_numpy()

        def read_genn(self, population):
            # **YUCK** I would like to implement this with a mixin
            # to reduce copy-paste but seemingly nanobind doesn't like this
            # Read from stream into GeNN-owned memory
            super().read_genn(population.extra_global_params["input"].view)

            # Copy data to device
            # **NOTE** this may be a NOP if CPU backend is used
            population.push_extra_global_param_to_device("input")

            return population.extra_global_params["input"].view

except:
    pass  # Ignore if drivers are not installed

try:

    class SpeckInput(ext.SpeckInput):
        def read(self):
            t = self.read_buffer()
            if USE_TORCH:
                return t.to_torch()
            else:
                return t.to_numpy()

        def read_genn(self, population):
            # **YUCK** I would like to implement this with a mixin
            # to reduce copy-paste but seemingly nanobind doesn't like this
            # Read from stream into GeNN-owned memory
            super().read_genn(population.extra_global_params["input"].view)

            # Copy data to device
            # **NOTE** this may be a NOP if CPU backend is used
            population.push_extra_global_param_to_device("input")

            return population.extra_global_params["input"].view

except Exception as e:
    print("NO SPECK", e)
    pass  # Ignore if Speck/ZMQ isn't installed
