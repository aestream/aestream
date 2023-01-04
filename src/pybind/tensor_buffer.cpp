#include "tensor_buffer.hpp"

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
// CUDA functions
int *alloc_memory_cuda(size_t buffer_size);
void free_memory_cuda(int *cuda_device_pointer);
void index_increment_cuda(torch::Tensor array, std::vector<int> events,
                          int *event_device_pointer);
#endif
// TensorBuffer constructor
TensorBuffer::TensorBuffer(py_size_t size, device_t device, size_t buffer_size)
#ifdef USE_TORCH
    : shape(size.vec()) {
  options_buffer = torch::TensorOptions()
                       .dtype(torch::kInt16)
                       .device(device)
                       .memory_format(c10::MemoryFormat::Contiguous)
                       .pinned_memory(true);
  options_copy = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  buffer1 = std::make_shared<tensor_t>(torch::zeros(size, options_buffer));
  buffer2 = std::make_shared<tensor_t>(torch::zeros(size, options_buffer));
#else
    : shape(size) {
  if (device != "cpu") {
    throw std::invalid_argument("Unknown device for Numpy tensor");
  }
  buffer1 = std::make_shared<cache_t>(cache_t(size[0] * size[1]));
  buffer2 = std::make_shared<cache_t>(cache_t(size[0] * size[1]));
#endif

#ifdef USE_CUDA
  cuda_device_pointer = alloc_memory_cuda(buffer_size);
  offset_buffer = std::vector<int>(buffer_size);
#endif
}

TensorBuffer::~TensorBuffer() {}

void TensorBuffer::set_buffer(uint16_t data[], int numbytes) {
  const auto length = numbytes >> 1;
  const std::lock_guard lock{buffer_lock};
#ifdef USE_CUDA
  if (buffer1->device().is_cuda()) {
    offset_buffer.clear();
    for (int i = 0; i < length; i = i + 2) {
      // Decode x, y
      const uint16_t y_coord = data[i] & 0x7FFF;
      const uint16_t x_coord = data[i + 1] & 0x7FFF;
      offset_buffer.push_back(shape[1] * x_coord + y_coord);
    }
    index_increment_cuda(*buffer1, offset_buffer, cuda_device_pointer);
    return;
  }
#endif
#ifdef USE_TORCH
  int16_t *array = buffer1->data_ptr<int16_t>();
#else
  float *array = buffer1->data();
#endif
  for (int i = 0; i < length; i = i + 2) {
    // Decode x, y
    const int16_t y_coord = data[i] & 0x7FFF;
    const int16_t x_coord = data[i + 1] & 0x7FFF;
    (*(array + shape[1] * x_coord + y_coord))++;
  }
}

void TensorBuffer::set_vector(std::vector<AEDAT::PolarityEvent> events) {
  const std::lock_guard lock{buffer_lock};
#ifdef USE_CUDA
  if (buffer1->device().is_cuda()) {
    for (size_t i = 0; i < events.size(); i++) {
      offset_buffer[i] = shape[1] * events[i].x + events[i].y;
    }
    index_increment_cuda(*buffer1, offset_buffer, cuda_device_pointer);
    return;
  }
#endif
#ifdef USE_TORCH
  int16_t *array = buffer1->data_ptr<int16_t>();
#else
  float *array = buffer1->data();
#endif
  for (auto event : events) {
    (*(array + shape[1] * event.x + event.y))++;
  }
}

tensor_t TensorBuffer::read() {
  // Swap out old pointer
  {
    const std::lock_guard lock{buffer_lock};
    buffer1.swap(buffer2);
  }
  // Copy and clean
#ifdef USE_TORCH
  tensor_t copy = buffer2->to(options_copy, false, true);
  buffer2->index_put_({torch::indexing::Slice()}, 0);
#else
  // Create a Python object that will free the allocated
  // memory when destroyed:
  // Thanks to https://stackoverflow.com/a/44682603
  // and
  // https://github.com/ssciwr/pybind11-numpy-example/blob/main/python/pybind11-numpy-example_python.cpp#L43
  tensor_t copy = tensor_t(buffer2->size(), buffer2->data());
  copy.resize(shape);
  copy.owndata();
  cache_t *array = new cache_t(shape[0] * shape[1]);
  buffer2.reset(array);
#endif
  return copy;
}