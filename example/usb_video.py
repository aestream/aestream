import time
import torch  # Torch is needed to import c10 (Core TENsor) context
from aestream import DVSInput
import sdl

# Define our camera resolution
resolution = (640, 480)

# Initialize our canvas
window, pixels = sdl.create_sdl_surface(*resolution)

# Start streaming from a DVS camera on USB 2:2
with DVSInput(resolution, device="cuda") as stream:
    while True:
        # Read a tensor (640, 480) tensor from the camera
        tensor = stream.read().cpu()
        # 1tensor = torch.randn(640, 480).float() + 2

        # Render pixels
        pixels[:] = sdl.events_to_bw(tensor)
        window.refresh()
        time.sleep(0.01)
