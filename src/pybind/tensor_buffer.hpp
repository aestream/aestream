#pragma once

#include <string>
#include <thread>
#include <vector>

#include "../aer.hpp"
#include "types.hpp"

#ifdef USE_TORCH
#include <torch/extension.h>
#include <torch/torch.h>
#else
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#endif

class TensorBuffer {
private:
  const std::vector<int64_t> shape;
  uint64_t current_timestamp = 0;
#ifdef USE_TORCH
  torch::TensorOptions options_buffer;
  torch::TensorOptions options_copy;
#endif

  std::mutex buffer_lock;
  std::shared_ptr<cache_t> buffer1;
  std::shared_ptr<cache_t> buffer2;
#ifdef USE_CUDA
  std::vector<int> offset_buffer;
  int *cuda_device_pointer;
#endif
public:
  TensorBuffer(py_size_t size, device_t device, size_t buffer_size);
  ~TensorBuffer();
  template <typename T> void assign_event(T *array, int16_t x, int16_t y);
  void set_buffer(uint16_t data[], int numbytes);
  void set_vector(std::vector<AER::Event> events);
  tensor_t read();
};