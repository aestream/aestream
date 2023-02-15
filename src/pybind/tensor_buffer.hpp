#pragma once

#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../aer.hpp"
#include "types.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

// struct nop
// {
//     template <typename T>
//     void operator() (T const &) const noexcept { }
// };

using tensor_t = nb::tensor<float, nb::shape<2, nb::any>>;
using buffer_t = float *;

class TensorBuffer {
private:
  const std::vector<int64_t> shape;
  uint64_t current_timestamp = 0;
  const std::string &device;

  std::mutex buffer_lock;
  buffer_t buffer1;
  buffer_t buffer2;
#ifdef USE_CUDA
  std::vector<int> offset_buffer;
  int *cuda_device_pointer;
#endif
public:
  TensorBuffer(py_size_t size, const std::string &device, size_t buffer_size);
  ~TensorBuffer();
  template <typename R> void assign_event(R *array, int16_t x, int16_t y);
  void set_buffer(uint16_t data[], int numbytes);
  void set_vector(std::vector<AER::Event> events);
  tensor_t read();
};