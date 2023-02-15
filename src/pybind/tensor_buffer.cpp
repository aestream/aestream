#include "tensor_buffer.hpp"

namespace nb = nanobind;

#ifdef USE_CUDA
void index_increment_cuda(float *array, std::vector<int> offsets,
                          int *event_device_pointer);
float *alloc_memory_cuda_float(size_t buffer_size);
int *alloc_memory_cuda_int(size_t buffer_size);
void free_memory_cuda(float *cuda_device_pointer);
#endif

inline buffer_t allocate_buffer(const size_t &length) {
#ifdef USE_CUDA
  return alloc_memory_cuda_float(length);
#endif
  return new float[length]{0};
}

// TensorBuffer constructor
TensorBuffer::TensorBuffer(py_size_t size, const std::string &device,
                           size_t buffer_size)
    : shape(size), device(device) {
#ifdef USE_CUDA
  if (device == "cuda") {
    alloc_memory_cuda_int(buffer_size);
    offset_buffer = std::vector<int>(buffer_size);
  }
#endif
  buffer1 = allocate_buffer(size[0] * size[1]);
  buffer2 = allocate_buffer(size[0] * size[1]);
}

TensorBuffer::~TensorBuffer() {}

void TensorBuffer::set_buffer(uint16_t data[], int numbytes) {
  const auto length = numbytes >> 1;
  const std::lock_guard lock{buffer_lock};
  // #ifdef USE_CUDA
  //   if (device == "cuda") {
  //     offset_buffer.clear();
  //     for (int i = 0; i < length; i = i + 2) {
  //       // Decode x, y
  //       const uint16_t y_coord = data[i] & 0x7FFF;
  //       const uint16_t x_coord = data[i + 1] & 0x7FFF;
  //       offset_buffer.push_back(shape[1] * x_coord + y_coord);
  //     }
  //     index_increment_cuda(buffer1.get(), offset_buffer,
  //     cuda_device_pointer); return;
  //   }
  // #endif
  // for (int i = 0; i < length; i = i + 2) {
  //   // Decode x, y
  //   const int16_t y_coord = data[i] & 0x7FFF;
  //   const int16_t x_coord = data[i + 1] & 0x7FFF;
  //   assign_event(buffer1.get(), x_coord, y_coord);
  // }
}

void TensorBuffer::set_vector(std::vector<AER::Event> events) {
  const std::lock_guard lock{buffer_lock};
  // #ifdef USE_CUDA
  //   if (device == "cuda") {
  //     for (size_t i = 0; i < events.size(); i++) {
  //       offset_buffer[i] = shape[1] * events[i].x + events[i].y;
  //     }
  //     index_increment_cuda(buffer1, offset_buffer, cuda_device_pointer);
  //     return;
  //   }
  // #endif
  //   for (auto event : events) {
  //     assign_event(buffer1, event.x, event.y);
  //   }
}

template <typename R>
inline void TensorBuffer::assign_event(R *array, int16_t x, int16_t y) {
  (*(array + shape[1] * x + y))++;
}

tensor_t TensorBuffer::read() {
  // Swap out old pointer
  {
    const std::lock_guard lock{buffer_lock};
    float *tmp = buffer1;
    buffer1 = buffer2;
    buffer2 = tmp;
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
  // Extract old pointer
  float *data = buffer2;
  // Replace old buffer
  buffer2 = allocate_buffer(shape[0] * shape[1]);
  // Return tensor
  const size_t s[2] = {shape[0], shape[1]};
  return tensor_t(data, 2, s);
}