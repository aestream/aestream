import time

import torch
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from aestream import DVSInput

## Example modified from: https://matplotlib.org/stable/tutorials/advanced/blitting.html

# Initialize our canvas
fig, ax = plt.subplots()
image = ax.imshow(torch.zeros(260, 346), cmap="gray", vmin=0, vmax=1)
plt.show(block=False)
plt.pause(0.1)
bg = fig.canvas.copy_from_bbox(fig.bbox)
ax.draw_artist(image)
fig.canvas.blit(fig.bbox)

# Start streaming from a DVS camera on USB 2:3
with DVSInput(2, 2, (640, 480)) as stream:
    while True:
        # Read a tensor (346, 260) tensor from the camera
        tensor = stream.read()

        # Redraw figure
        fig.canvas.restore_region(bg)
        image.set_data(tensor.T.numpy())
        ax.draw_artist(image)
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()

        # Pause to only loop 10 times per second
        plt.pause(0.01)
