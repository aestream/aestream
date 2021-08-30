#pragma once
#include "aedat.hpp"
#include "aedat4.hpp"

#include <iostream>
#include <sys/types.h>
#include <torch/script.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

torch::Tensor convert_polarity_events(
    std::vector<AEDAT::PolarityEvent> &polarity_events,
    const std::vector<int64_t> &tensor_size = std::vector<int64_t>());

std::vector<torch::Tensor>
convert_polarity(std::vector<AEDAT::PolarityEvent> &polarity_events,
                 const int64_t window_size,
                 const int64_t window_step,
                 const std::vector<double> &scale,
                 const std::vector<int64_t> &image_dimensions) {