import os, sys
import torch
import dvs2tensor
import argparse

def sparse_generator_func(packet_size, buffer_size, ID):
    """
    Generates sparse data from DVS camera data
    """
    dvsdataconv = dvs2tensor.DVSDataConv(packet_size, buffer_size)
    dvsdataconv.connect2camera(ID)
    dvsdataconv.startdatastream()

    try:
        while True:
            yield dvsdataconv.update()
    except GeneratorExit:
        exitcode = dvsdataconv.stopdatastream()


if __name__ == "__main__":
    # Get input arguments from shell
    parser = argparse.ArgumentParser("Stream sparse tensor data")

    # General Configs
    parser.add_argument("--events", default=1000, type=int, help="Specify number of events")
    parser.add_argument("--id", default=1, type=int, help="Specify camera id")

    # Event configs
    parser.add_argument("--packet_size", default=1000, type=int, help="Specify interval size")
    parser.add_argument("--buffer", default=512, type=int, help="Specify size of buffer")

    # Get arguments
    args = parser.parse_args()

    gen = sparse_generator_func(args.packet_size, args.buffer, args.id)

    i = 0
    # Iterate over generator
    for sparse_tensor in gen:
        print("First part of tensor: " + str(sparse_tensor))
        print("Tensor size: " + str(sparse_tensor.shape))
        i+= 1
        if i == args.events:
            break