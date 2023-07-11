import time
import sdl

from scipy.ndimage import zoom

from aestream import SpeckInput

# Initialize our canvas
window, pixels = sdl.create_sdl_surface(512, 512)

# Start streaming from a DVS camera on USB 2:2
with SpeckInput() as stream:
    while True:
        # Read a tensor (640, 480) tensor from the camera
        tensor = stream.read()

        # Zoom to (512, 512)
        tensor = zoom(tensor, 4, order=0)

        # Render pixels
        pixels[:] = sdl.events_to_bw(tensor)
        window.refresh()
        time.sleep(0.01)
