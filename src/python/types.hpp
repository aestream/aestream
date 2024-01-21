#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <nanobind/stl/vector.h>

namespace nb = nanobind;

using py_size_t = std::vector<size_t>;

enum Backend { GeNN, Jax, Numpy, Torch };

enum Camera { Inivation, Prophesee };