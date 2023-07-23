#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "../cpp/aer.hpp"
#include "../cpp/generator.hpp"

#include "tensor_buffer.hpp"

class TensorIterator {
private:
  const py_size_t shape;
  const size_t time_window;

  uint64_t current_timestamp = 0;

public:
  Generator<AER::Event> &generator;

  TensorIterator(Generator<AER::Event> &generator, py_size_t shape,
                 size_t time_window);
  template <typename T> void assign_event(T *array, int16_t x, int16_t y);
  float * next();
};