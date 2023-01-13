import time

import numpy
from aestream import FileInput


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
            events += frame.sum()
    assert events == 539136
