# Import the deep learning library, PyTorch
import torch

# Import the spiking neural network library, Norse
import norse

# Import the DVS camera streaming library, AEstream
from aestream import USBInput

# Initialize our canvas
from sdl import create_sdl_surface, events_to_bw

window, pixels = create_sdl_surface(640 * 3, 480)

# Create horizontal and vertical edge detectors
kernel_size = 9
gaussian = torch.sigmoid(torch.linspace(-10, 10, kernel_size + 1))
kernel = (gaussian.diff() - 0.14).repeat(kernel_size, 1)
kernels = torch.stack((kernel, kernel.T))
convolution = torch.nn.Conv2d(1, 2, kernel_size, padding=12, bias=False, dilation=3)
convolution.weight = torch.nn.Parameter(kernels.unsqueeze(1))

# Create Norse network
# - One refractory cell to inhibit pixels
# - One convolutional edge-detection layer
net = norse.torch.SequentialState(
    norse.torch.LIFRefracCell(),
    convolution,
)
state = None  # Start with empty state

try:
    # Start streaming from a DVS camera on USB 2:2 and put them on the CPU
    with USBInput((640, 480)) as stream:
        while True:  # Loop forever
            # Read a tensor (640, 480) tensor from the camera
            tensor = stream.read("torch")
            # Run the tensor through the network, while updating the state
            with torch.inference_mode():
                filtered, state = net(tensor.view(1, 1, 640, 480), state)

            # Render tensors
            pixels[0:640] = events_to_bw(tensor)  # Input events
            pixels[640 : 640 * 2] = events_to_bw(filtered[0, 0])  # First channel
            pixels[640 * 2 : 640 * 3] = events_to_bw(filtered[0, 1])  # Second channel
            window.refresh()

finally:
    window.close()
