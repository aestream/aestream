import numpy as np
from pygenn.genn_model import create_custom_neuron_class

genn_input_model = create_custom_neuron_class(
    "genn_input",
    extra_global_params=[("input", "uint32_t*")],
    threshold_condition_code="""
    $(input)[$(id) / 32] & (1 << ($(id) % 32))
    """,
    is_auto_refractory_required=False,
)


def add_input(model, name, resolution):
    # Calculate total number of neurons in input
    # and hence required words to encode
    num_neurons = np.prod(resolution)
    num_words = (num_neurons + 31) // 32

    # Add input and initialise extra global parameter
    pop = model.add_neuron_population("input", num_neurons, genn_input_model, {}, {})
    pop.set_extra_global_param("input", np.empty(num_words, dtype=np.uint32))

    return pop
