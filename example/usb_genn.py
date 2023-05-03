import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

from aestream import USBInput
from pygenn.genn_model import GeNNModel
from pygenn.genn_model import create_custom_neuron_class

genn_input_model = create_custom_neuron_class(
    "genn_input",
    extra_global_params=[("input", "uint32_t*")],
    
    threshold_condition_code="""
    $(input)[$(id) / 32] & (1 << ($(id) % 32))
    """,
    is_auto_refractory_required=False)


model = GeNNModel("float", "usb_genn")
pop = model.add_neuron_population("input", 640 * 480, genn_input_model, {}, {})
pop.set_extra_global_param("input", np.empty(9600, dtype=np.uint32))
pop.spike_recording_enabled = True

model.build()
model.load(num_recording_timesteps=100)

# Connect to a USB camera, receiving tensors of shape (640, 480)
# By default, we send the tensors to the CPU
#   - if you have a GPU, try changing this to "cuda"
with USBInput((640, 480), device="genn") as stream:
    # Loop forever
    while True:
        for i in range(100):
            stream.read_genn(pop.extra_global_params["input"].view)
            pop.push_extra_global_param_to_device("input")
        
            model.step_time()
        
        model.pull_recording_buffers_from_device()
        spike_times, spike_ids = pop.spike_recording_data
        
        fig, axis = plt.subplots()
        spike_x = spike_ids % 640
        spike_y = spike_ids // 640
        axis.scatter(spike_x, spike_y, s=1)
        plt.show()
