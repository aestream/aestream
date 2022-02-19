import datetime
import multiprocessing
import socket
import time

import torch
from aestream import UDPInput

def stream_fake_data():
    data = b"\x0F\x00\xDA\x00"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(data, ("127.0.0.1", 3333))
    sock.close()
    

def start_stream():
    p = multiprocessing.Process(target=stream_fake_data)
    p.start()
    return p

def test_udp():
    with UDPInput(torch.Size((640, 480))) as stream:
        start_stream() # Start streaming from file

        interval = 0.5
        t_0 = time.time()
        while True:
            if t_0 + interval <= time.time():
                frame = stream.read()
                break
    
    assert torch.eq(frame[218, 15], 1)

if __name__ == "__main__":
    test_udp()