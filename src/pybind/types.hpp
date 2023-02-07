#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

using cache_t = float *;
using tensor_t = py::array_t<float, py::array::c_style | py::array::forcecast>;
using py_size_t = std::vector<py::ssize_t>;
