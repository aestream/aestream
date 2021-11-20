#pragma once

#include <chrono>
#include <torch/torch.h>

#include "aedat.hpp"
#include "generator.hpp"

Generator<torch::Tensor>
sparse_tensor_generator(Generator<AEDAT::PolarityEvent>& event_generator,
                       std::chrono::duration<double, std::micro> event_window);

// Generator<torch::Tensor> sparse_tensor_generator(Generator <
//  AEDAT::PolarityEvent);