#include "tensor_buffer.hpp"
#include <iostream>

namespace nb = nanobind;

// TensorBuffer constructor
TensorBuffer::TensorBuffer(py_size_t size, const std::string &device,
                           size_t buffer_size)
    : shape(size), device(device) {
#ifdef USE_CUDA
  cuda_device_pointer = alloc_memory_cuda<int>(buffer_size);
  offset_buffer = std::vector<int>(buffer_size);
  if (device == "cuda") {
    buffer1 =
        std::make_shared<cache_t>(alloc_memory_cuda<float>(size[0] * size[1]));
    buffer2 =
        std::make_shared<cache_t>(alloc_memory_cuda<float>(size[0] * size[1]));
  } else {
#endif
    buffer1 = std::make_unique<cache_t>(size[0] * size[1]);
    buffer2 = std::make_unique<cache_t>(size[0] * size[1]);
#ifdef USE_CUDA
  }
#endif
}

TensorBuffer::~TensorBuffer() {}

void TensorBuffer::set_buffer(uint16_t data[], int numbytes) {
  const auto length = numbytes >> 1;
  const std::lock_guard lock{buffer_lock};
#ifdef USE_CUDA
  if (device == "cuda") {
    offset_buffer.clear();
    for (int i = 0; i < length; i = i + 2) {
      // Decode x, y
      const uint16_t y_coord = data[i] & 0x7FFF;
      const uint16_t x_coord = data[i + 1] & 0x7FFF;
      offset_buffer.push_back(shape[1] * x_coord + y_coord);
    }
    index_increment_cuda<float>(buffer1.get(), offset_buffer,
                                cuda_device_pointer);
    return;
  }
#endif
  for (int i = 0; i < length; i = i + 2) {
    // Decode x, y
    const int16_t y_coord = data[i] & 0x7FFF;
    const int16_t x_coord = data[i + 1] & 0x7FFF;
    assign_event(buffer1.get(), x_coord, y_coord);
  }
}

void TensorBuffer::set_vector(std::vector<AER::Event> events) {
  const std::lock_guard lock{buffer_lock};
#ifdef USE_CUDA
  if (device == "cuda") {
    for (size_t i = 0; i < events.size(); i++) {
      offset_buffer[i] = shape[1] * events[i].x + events[i].y;
    }
    index_increment_cuda<float>(*buffer1.get(), offset_buffer,
                                cuda_device_pointer);
    return;
  }
#endif
  for (auto event : events) {
    assign_event(buffer1.get(), event.x, event.y);
  }
}

template <typename T>
inline void TensorBuffer::assign_event(T *array, int16_t x, int16_t y) {
  (*(array + shape[1] * x + y))++;
}

tensor_t TensorBuffer::read() {
  // Swap out old pointer
  {
    const std::lock_guard lock{buffer_lock};
    buffer1.swap(buffer2);
  }
  // Copy and clean
  // #ifdef USE_CUDA
  //   tensor_t copy = tensor_t(shape[0] * shape[1], *buffer2.get());
  //   std::cout << "Bob" << std::endl;
  //   free_memory_cuda<float>(*buffer2.get());
  //   buffer2 = std::make_shared<cache_t>(alloc_memory_cuda<float>(shape[0] *
  //   shape[1]));
  // #else
  //   // Create a Python object that will free the allocated
  //   // memory when destroyed:
  //   // Thanks to https://stackoverflow.com/a/44682603
  //   // and
  //   //
  //   https://github.com/ssciwr/pybind11-numpy-example/blob/main/python/pybind11-numpy-example_python.cpp#L43
  //   tensor_t copy = tensor_t(shape[0] * shape[1], buffer2.get());
  //   copy.resize(shape);
  //   copy.owndata();
  //   cache_t *array = new cache_t(shape[0] * shape[1]);
  //   buffer2.reset(array);
  // #endif
  float *data = buffer2.get();
  const size_t s[2] = {shape[0], shape[1]};
  return tensor_t(data, 2, s);
}