#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#ifdef USE_TORCH
#include <torch.h>
using cache_t = at::Tensor;
using device_t = torch::Device;
using tensor_t = at::Tensor;
using py_size_t = torch::IntArrayRef;
#else
using cache_t = std::vector<float>;
using device_t = std::string;
using tensor_t = py::array_t<float, py::array::c_style | py::array::forcecast>;
using py_size_t = std::vector<py::ssize_t>;
#endif