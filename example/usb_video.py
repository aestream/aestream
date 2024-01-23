import time
import aestream
import sdl

# Define our camera resolution
resolution = (640, 480)

# Initialize our canvas
window, pixels = sdl.create_sdl_surface(*resolution)

# Start streaming from a DVS camera on USB 2:2
# To enfore camera type, use USBInput(resolution, camera=aestream.Prophesee)
#                        or  USBInput(resolution, camera=aestream.Inivation)
with aestream.USBInput(resolution) as stream:
    while True:
        # Read a tensor (640, 480) tensor from the camera
        tensor = stream.read()

        # Render pixels
        pixels[:] = sdl.events_to_bw(tensor)
        window.refresh()
        time.sleep(0.01)
