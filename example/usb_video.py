from aestream import DVSInput
from example import sdl

## Example modified from: https://matplotlib.org/stable/tutorials/advanced/blitting.html

# Initialize our canvas
window, pixels = sdl.create_sdl_surface(640, 480)

# Start streaming from a DVS camera on USB 2:2
with DVSInput(2, 2, (640, 480)) as stream:
    while True:
        # Read a tensor (640, 480) tensor from the camera
        tensor = stream.read()

        # Render pixels
        pixels[0:640] =  sdl.events_to_bw(tensor)
