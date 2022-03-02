import time

import torch
import norse.torch as snn
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from aestream import DVSInput

## Example modified from: https://matplotlib.org/stable/tutorials/advanced/blitting.html

# Initialize our canvas
fig, (ax1, ax2) = plt.subplots(1, 2)
image1 = ax1.imshow(torch.zeros(260, 346), cmap="gray", vmin=0, vmax=1)
image2 = ax2.imshow(torch.zeros(260, 346), cmap="gray", vmin=0, vmax=2)
plt.show(block=False)
plt.pause(0.1)
bg = fig.canvas.copy_from_bbox(fig.bbox)
ax1.draw_artist(image1)
ax2.draw_artist(image2)
fig.canvas.blit(fig.bbox)

# Initialize PyTorch network
net = snn.LICell().cuda()
state = None

# Start streaming from a DVS camera on USB 2:8 and put them on the GPU
try:
    with DVSInput(2, 8, (346, 260), device="cuda") as stream:
        while True:
            # Read a tensor (346, 260) tensor from the camera
            tensor = stream.read()
            with torch.inference_mode():
                filtered, state = net(tensor.view(1, 1, 346, 260), state)

            # Redraw figure
            fig.canvas.restore_region(bg)
            image1.set_data(tensor.T.cpu().numpy())
            image2.set_data(filtered.squeeze().T.cpu().numpy())
            ax1.draw_artist(image1)
            ax2.draw_artist(image2)
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()

            # Pause to only loop 10 times per second
            plt.pause(0.01)
except Exception as e:
    print("Error", e)
