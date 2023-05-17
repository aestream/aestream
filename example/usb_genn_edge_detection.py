import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

from aestream import USBInput
from aestream import genn
from pygenn.genn_model import GeNNModel
from pygenn.genn_model import init_toeplitz_connectivity
import sdl

WIDTH = 640
HEIGHT = 480
RESOLUTION = (WIDTH, HEIGHT, 1)
NUM_TIMESTEPS_PER_FRAME = 16
KERNEL_SIZE = 9

model = GeNNModel("float", "usb_genn_edge_detection")
model.dT = 1.0

# Add input
input_pop = genn.add_input(model, "input", RESOLUTION)

edge_params = {
    "C": 1.0, "TauM": 20.0,
    "Vrest": 0.0, "Vreset": 0.0, "Vthresh": 5.0,
    "Ioffset": 0.0, "TauRefrac": 2}
edge_pop = model.add_neuron_population("edge", 160 * 120 * 2, "LIF", 
                                       edge_params, {"V": 0.0, "RefracTime": 0.0})
edge_pop.spike_recording_enabled = True

avg_pool_conv_params = {
    "conv_kh": KERNEL_SIZE, "conv_kw": KERNEL_SIZE,
    "pool_kh": 4, "pool_kw": 4,
    "pool_sh": 4, "pool_sw": 4,
    "pool_ih": 480, "pool_iw": 640, "pool_ic": 1,
    "conv_oh": 120, "conv_ow": 160, "conv_oc": 2}

# Build 3x3 vertical edge detection kernel
gaussian = expit(np.linspace(-10, 10, KERNEL_SIZE + 1))
kernel = np.tile((np.diff(gaussian) - 0.14), (KERNEL_SIZE, 1))

model.add_synapse_population("input_edge", "TOEPLITZ_KERNELG", 0,
                             input_pop, edge_pop,
                             "StaticPulse", {}, {"g": np.stack((kernel, kernel.T), 2).flatten()}, {}, {},
                             "DeltaCurr", {}, {},
                             init_toeplitz_connectivity("AvgPoolConv2D", avg_pool_conv_params))
                             
model.build()
model.load(num_recording_timesteps=NUM_TIMESTEPS_PER_FRAME)

# Initialize our canvas
window, pixels = sdl.create_sdl_surface(640 + 160, 480)

# Connect to a USB camera, receiving tensors of shape (640, 480)
in_view = input_pop.extra_global_params["input"].view
in_data = np.zeros(in_view.shape, np.uint32)
with USBInput(RESOLUTION, device="genn") as stream:
    # Loop forever
    while True:
        # Run one frames worth of timesteps
        in_data[:] = 0
        for i in range(NUM_TIMESTEPS_PER_FRAME):
            in_data = np.bitwise_or(in_data, stream.read_genn(input_pop))

            model.step_time()

        # Download one frame of edge detector spikes from device
        model.pull_recording_buffers_from_device()
        _, edge_spike_ids = edge_pop.spike_recording_data
        
        # Decode
        edge_y, edge_x, edge_c = np.unravel_index(edge_spike_ids, (120, 160, 2))
        
        # Decode input spike IDs
        in_spike_ids = np.unpackbits(in_data.view(np.uint8), bitorder="little")
        in_y, in_x = np.unravel_index(np.where(in_spike_ids == 1)[0], (HEIGHT, WIDTH))
        
        # Clear pixels
        pixels[:] = 0
        
        # Show input pixels in red
        pixels[in_x, in_y] = (255 << 16)
        
        # Show horizontal and vertical pixels in blue and green
        pixels[edge_x + 640, edge_y + (edge_c * 240)] = (255 << (edge_c * 8))
        
        window.refresh()
        
