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
