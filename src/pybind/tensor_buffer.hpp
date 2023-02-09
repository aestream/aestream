#pragma once

#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../aer.hpp"
#include "types.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#ifdef USE_CUDA
#include "tensor_buffer_kernel.h"
#endif

class TensorBuffer {
private:
  const std::vector<int64_t> shape;
  uint64_t current_timestamp = 0;
  const std::string &device;
#ifdef USE_TORCH
  torch::TensorOptions options_buffer;
  torch::TensorOptions options_copy;
#endif

  std::mutex buffer_lock;
  std::unique_ptr<cache_t> buffer1;
  std::unique_ptr<cache_t> buffer2;
#ifdef USE_CUDA
  std::vector<int> offset_buffer;
  int *cuda_device_pointer;
#endif
public:
  TensorBuffer(py_size_t size, const std::string& device, size_t buffer_size);
  ~TensorBuffer();
  template <typename T> void assign_event(T *array, int16_t x, int16_t y);
  void set_buffer(uint16_t data[], int numbytes);
  void set_vector(std::vector<AER::Event> events);
  tensor_t read();
};