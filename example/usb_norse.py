# Import the deep learning library, PyTorch
import torch

# Import the spiking neural network library, Norse
import norse

# Import the DVS camera streaming library, AEstream
from aestream import USBInput

# Initialize our canvas
from sdl import create_sdl_surface, events_to_bw

window, pixels = create_sdl_surface(640 * 2, 480)

# Create a simple Norse leaky integrator
net = norse.torch.LICell(p=norse.torch.LIParameters(tau_syn_inv=100))
state = None  # Start with empty state

try:
    # Start streaming from a DVS camera on USB 2:2 and put them on the CPU
    with USBInput((640, 480), device="cuda") as stream:
        while True:  # Loop forever
            # Read a tensor (640, 480) tensor from the camera
            tensor = stream.read("torch").cpu()
            # Run the tensor through the network, while updating the state
            with torch.inference_mode():
                filtered, state = net(tensor.view(1, 640, 480), state)

            # Render tensors
            pixels[0:640] = events_to_bw(tensor)  # Input events
            pixels[640 : 640 * 2] = events_to_bw(filtered[0])  # First channel
            window.refresh()

finally:
    window.close()
