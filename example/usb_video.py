import time
from aestream import USBInput
import sdl

# Define our camera resolution
resolution = (640, 480)

# Initialize our canvas
window, pixels = sdl.create_sdl_surface(*resolution)

# Start streaming from a DVS camera on USB 2:2
with USBInput(resolution) as stream:
    while True:
        # Read a tensor (640, 480) tensor from the camera
        tensor = stream.read()
        # 1tensor = torch.randn(640, 480).float() + 2

        # Render pixels
        pixels[:] = sdl.events_to_bw(tensor)
        window.refresh()
        time.sleep(0.01)
