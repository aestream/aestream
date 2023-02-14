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
#include <cuda.h>
template <typename scalar_t>
void index_increment_cuda(scalar_t *array, std::vector<int> offsets,
                          int *event_device_pointer);

template <typename scalar_t> scalar_t *alloc_memory_cuda(size_t buffer_size);

template <typename scalar_t>
void free_memory_cuda(scalar_t *cuda_device_pointer);

using tensor_t =
    nb::tensor<nb::pytorch, float, nb::shape<2, nb::any>, nb::device::cuda>;
#else
using tensor_t = nb::tensor<nb::numpy, float, nb::shape<2, nb::any>>;
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
  TensorBuffer(py_size_t size, const std::string &device, size_t buffer_size);
  ~TensorBuffer();
  template <typename R> void assign_event(R *array, int16_t x, int16_t y);
  void set_buffer(uint16_t data[], int numbytes);
  void set_vector(std::vector<AER::Event> events);
  tensor_t read();
};