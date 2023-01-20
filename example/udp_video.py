import time

import torch
from aestream import UDPInput

# Start a stream, receiving tensors of shape (640, 480)
# By default, we listen on port 3333 and send the tensors to the CPU
# The variable stream can now be `.read()` whenever a tensor is desired
with UDPInput((640, 480)) as stream:

    # Create a tensor list
    images = []

    # Loop forever
    for i in range(4000):
        # Grab a tensor of events
        frame = stream.read()

        # Add it to our list
        images.append(frame)

        # Sleep at least 1 ms
        time.sleep(0.001)

    # Save the tensor to a binary file
    torch.save(torch.stack(images), "aestream.dat")
