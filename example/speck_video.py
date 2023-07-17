import time
import sdl

from scipy.ndimage import zoom

from aestream import SpeckInput

# Initialize our canvas
window, pixels = sdl.create_sdl_surface(512, 512)

# Start streaming from the SynSense Speck chip
with SpeckInput() as stream:
    while True:
        # Read a tensor (128, 128) tensor from the camera
        tensor = stream.read()

        # Zoom to (512, 512)
        tensor = zoom(tensor, 4, order=0)

        # Render pixels
        pixels[:] = sdl.events_to_bw(tensor)
        window.refresh()
        time.sleep(0.01)
