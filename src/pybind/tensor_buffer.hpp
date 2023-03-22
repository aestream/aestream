#pragma once

#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../aer.hpp"
#include "types.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/unique_ptr.h>

#ifdef USE_CUDA
void index_increment_cuda(float *array, int *offset_pointer, size_t indices,
                            int *event_device_pointer);
void *alloc_memory_cuda(size_t buffer_size, size_t bytes);
void free_memory_cuda(void *cuda_device_pointer);
#endif
template <typename scalar_t> struct BufferDeleter {
  void operator()(scalar_t *ptr) {
#ifdef USE_CUDA
    free_memory_cuda(static_cast<void *>(ptr));
#else
    delete ptr;
#endif
  }
};

using tensor_numpy = nb::ndarray<nb::numpy, float, nb::shape<2, nb::any>>;
using tensor_torch = nb::ndarray<nb::pytorch, float, nb::shape<2, nb::any>>;
using buffer_t = std::unique_ptr<float[], BufferDeleter<float>>;
using index_t = std::unique_ptr<int[], BufferDeleter<int>>;

struct BufferPointer {
  BufferPointer(buffer_t data, const std::vector<int64_t> &shape,
                std::string device);
  tensor_numpy to_numpy();
  tensor_torch to_torch();

private:
  std::string device;
  const std::vector<int64_t> &shape;
  buffer_t data;
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
  index_t cuda_buffer;
#endif
public:
  TensorBuffer(py_size_t size, std::string device, size_t buffer_size);
  ~TensorBuffer();
  template <typename R> void assign_event(R *array, int16_t x, int16_t y);
  void set_buffer(uint16_t data[], int numbytes);
  void set_vector(std::vector<AER::Event> events);
  BufferPointer read();
};