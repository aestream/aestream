import time
import pytest

import numpy as np
from aestream import Event, FileInput

from . import _has_cuda_torch, _has_torch

class namespace:
    pass

def test_load_dat():
    f = FileInput("example/sample.dat", shape=(600, 400))
    buf = f.load()

    assert len(buf) == 539481
    assert buf[0]["timestamp"] == 0
    assert buf[0]["x"] == 237
    assert buf[0]["y"] == 121
    assert buf[0]["polarity"] == True


def test_stream_dat():
    with FileInput(
        filename="example/sample.dat", shape=(600, 400), ignore_time=True
    ) as stream:
        time.sleep(0.3)
        interval = 0.5
        t_0 = time.time()
        events = 0
        while True:
            if time.time() > t_0 + interval:
                break
            frame = stream.read()
            if _has_torch():
                import torch
                assert isinstance(frame, torch.Tensor)
            else:
                assert isinstance(frame, np.ndarray)
            events += frame.sum()
    assert events == 539136

@pytest.mark.skipif(not _has_cuda_torch(), reason="Torch-gpu is not installed")
def test_stream_dat_torch_cuda():
    import torch
    with FileInput(
        filename="example/sample.dat", shape=(600, 400), ignore_time=True, device="cuda"
    ) as stream:
        time.sleep(1)
        interval = 0.4
        t_0 = time.time()
        events = 0
        while True:
            if time.time() > t_0 + interval:
                break
            frame = stream.read()
            assert isinstance(frame, torch.Tensor)
            assert frame.device.type == "cuda"
            events += frame.sum()
            time.sleep(0.1)
        events += stream.read().sum()
    assert events == 539136
