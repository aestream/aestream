import torch
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from aestream import USBInput

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
net = torch.nn.Conv2d(1, 3, 5, padding=1, bias=False)
normal = torch.distributions.Normal(0, 1)
gaussian = normal
net.weight = torch.nn.Parameter(
    torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
)

# Start streaming from a DVS camera and put them on the GPU
with USBInput((640, 480), device="cuda") as stream:
    try:
        while True:
            # Read a tensor (346, 260) tensor from the camera
            tensor = stream.read("torch")
            with torch.inference_mode():
                filtered = net(tensor.view(1, 640, 480))

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
