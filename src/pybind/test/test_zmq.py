import pyzmq


class ZmqServer():
    
    def __init__(self):
        c = pyzmq.Context()
        self.s = c.socket(pyzmq.PUB)
        self.s.bind("tcp://*:5555")

    def close(self):
        self.s.close() 


