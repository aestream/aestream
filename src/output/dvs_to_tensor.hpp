#pragma once

#include <chrono>
#include <torch/torch.h>

#include "../aedat.hpp"
#include "../generator.hpp"

torch::Tensor
convert_polarity_events(std::vector<AEDAT::PolarityEvent> &polarity_events,
                        const torch::IntArrayRef &shape = {},
                        const torch::Device &device = torch::DeviceType::CPU);

Generator<torch::Tensor>
sparse_tensor_generator(Generator<AEDAT::PolarityEvent> &event_generator,
                        std::chrono::duration<double, std::micro> event_window,
                        const torch::IntArrayRef shape = torch::IntArrayRef({}),
                        const torch::Device &device = torch::DeviceType::CPU);
