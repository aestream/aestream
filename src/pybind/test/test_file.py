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
        while True:
            if time.time() > t_0 + interval:
                break
            frame = stream.read()
            if _has_torch():
                import torch
                assert isinstance(frame, torch.Tensor)
            else:
                assert isinstance(frame, numpy.ndarray)
            events += frame.sum()
    assert events == 539136


@pytest.mark.skipif(not _has_cuda_torch(), reason="Torch-gpu is not installed")
def test_read_dat_torch_cuda():
    import torch
    with FileInput(
        filename="example/sample.dat", shape=(600, 400), ignore_time=True, device="cuda"
    ) as stream:
        frame = stream.read()
        interval = 0.2
        t_0 = time.time()
        events = 0
        while True:
            time.sleep(0.05)
            if time.time() > t_0 + interval:
                break
            frame = stream.read()
            assert isinstance(frame, torch.Tensor)
            assert frame.device.type == "cuda"
            print(frame.sum())
            events += frame.sum()
    assert events == 539136
