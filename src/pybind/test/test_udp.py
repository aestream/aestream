import multiprocessing
import socket
import time
import pytest

import numpy
from aestream import UDPInput

from . import _has_cuda_torch


def stream_fake_data(port):
    data = b"\x0F\x00\xDA\x00"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(data, ("127.0.0.1", port))
    sock.close()


def start_stream(port):
    p = multiprocessing.Process(target=stream_fake_data, args=(port,))
    p.start()
    return p


def test_udp():
    with UDPInput((640, 480), device="cpu") as stream:
        start_stream(3333)  # Start streaming from file

        interval = 0.5
        t_0 = time.time()
        while True:
            if t_0 + interval <= time.time():
                frame = stream.read()
                break
    assert numpy.equal(frame[218, 15], 1)


@pytest.mark.skipif(not _has_cuda_torch(), reason="Torch-gpu is not installed")
def test_udp_gpu():
    import torch

    if torch.has_cuda:
        with UDPInput((640, 480), device="cuda", port=3334) as stream:
            start_stream(3334)  # Start streaming from file

            interval = 0.5
            time.sleep(0.5)
            t_0 = time.time()
            while True:
                if t_0 + interval <= time.time():
                    frame = stream.read()
                    print(frame.argmax(0).argmax(), frame.argmax(1).argmax())
                    break

        assert torch.eq(frame[218, 15], 1)


if __name__ == "__main__":
    test_udp()
    test_udp_gpu()
