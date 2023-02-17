import time
import pytest

import numpy
from aestream import FileInput

from . import _has_cuda_torch, _has_torch


def test_read_dat():
    with FileInput(
        filename="example/sample.dat", shape=(600, 400), ignore_time=True
    ) as stream:
        interval = 0.5
        t_0 = time.time()
        events = 0
        time.sleep(0.1)
        while True:
            if time.time() > t_0 + interval:
                break
            frame = stream.read()
            if _has_torch:
                import torch
                assert isinstance(frame, torch.Tensor)
            else:
                assert isinstance(frame, numpy.array)
            events += frame.sum()
    assert events == 539136


@pytest.mark.skipif(not _has_cuda_torch, reason="Torch-gpu is not installed")
def test_read_dat_torch_cuda():
    import torch
    with FileInput(
        filename="example/sample.dat", shape=(600, 400), ignore_time=True, device="cuda"
    ) as stream:
        interval = 0.5
        t_0 = time.time()
        events = 0
        time.sleep(0.1)
        while True:
            if time.time() > t_0 + interval:
                break
            frame = stream.read()
            assert isinstance(frame, torch.Tensor)
            assert frame.device.type == "cuda"
            events += frame.sum()
    assert events == 539136
