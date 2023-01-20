#pragma once

#include <iostream>
#ifdef USE_TORCH
#include <torch/extension.h>
#endif

#include "aedat.hpp"
#include "aedat4.hpp"

// torch::Tensor convert_polarity_events(
//     std::vector<AEDAT::PolarityEvent> &polarity_events,
//     const std::vector<int64_t> &tensor_size = std::vector<int64_t>());

// std::vector<torch::Tensor>
// convert_polarity(std::vector<AEDAT::PolarityEvent> &polarity_events,
//                  const int64_t window_size, const int64_t window_step,
//                  const std::vector<double> &scale,
//                  const std::vector<int64_t> &image_dimensions);