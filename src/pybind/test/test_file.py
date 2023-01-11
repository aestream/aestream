import time

import numpy
from aestream import FileInput


def test_read_dat():
    with FileInput(
        filename="example/sample.aedat4", shape=(600, 400), ignore_time=True
    ) as stream:
        interval = 0.5
        t_0 = time.time()
        time.sleep(0.1)
        while True:
            # if t_0 + interval <= time.time():
            frame = stream.read()
            print(frame.sum())
            time.sleep(0.1)
    # assert numpy.equal(frame[218, 15], 1)
