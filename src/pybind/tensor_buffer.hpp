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
void index_increment_cuda(float *array, std::vector<int> offsets,
                          int *event_device_pointer);
float *alloc_memory_cuda_float(size_t buffer_size);
int *alloc_memory_cuda_int(size_t buffer_size);
void free_memory_cuda(float *cuda_device_pointer);
#endif
struct BufferDeleter {
  void operator()(float *ptr) {
#ifdef USE_CUDA
    free_memory_cuda(ptr);
#else
    delete ptr;
#endif
  }
};

using tensor_numpy = nb::tensor<nb::numpy, float, nb::shape<2, nb::any>>;
using tensor_torch = nb::tensor<nb::pytorch, float, nb::shape<2, nb::any>>;
using buffer_t = std::unique_ptr<float[], BufferDeleter>;

struct BufferPointer {
  BufferPointer(buffer_t data, const std::vector<int64_t> &shape, std::string device);
  tensor_numpy to_numpy();
  tensor_torch to_torch();

private:
  buffer_t data;
  std::string device;
  const std::vector<int64_t> &shape;
};

class TensorBuffer {
private:
  const std::vector<int64_t> shape;
  uint64_t current_timestamp = 0;
  std::string device;

  std::mutex buffer_lock;
  buffer_t buffer1;
  buffer_t buffer2;
#ifdef USE_CUDA
  std::vector<int> offset_buffer;
  int *cuda_device_pointer;
#endif
public:
  TensorBuffer(py_size_t size, std::string device, size_t buffer_size);
  ~TensorBuffer();
  template <typename R> void assign_event(R *array, int16_t x, int16_t y);
  void set_buffer(uint16_t data[], int numbytes);
  void set_vector(std::vector<AER::Event> events);
  BufferPointer read();
};