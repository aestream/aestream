#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#include <nanobind/stl/vector.h>

namespace nb = nanobind;

using cache_t = float[];
// #ifdef USE_CUDA
// using tensor_t = nb::tensor<nb::pytorch, float, nb::shape<2, nb::any>>;
// #else
// using tensor_t = nb::tensor<nb::numpy, float, nb::shape<2, nb::any>>;
// using tensor_t = nb::tensor<float>;
// #endif
using py_size_t = std::vector<ssize_t>;
