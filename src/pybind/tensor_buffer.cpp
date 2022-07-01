#include <string>
#include <thread>
#include <vector>

#include "../aedat.hpp"
#include "tensor_buffer.hpp"

#include <torch/extension.h>
#include <torch/torch.h>

TensorBuffer::TensorBuffer(torch::IntArrayRef size, torch::Device device)
    : shape(size.vec()) {
  options_buffer = torch::TensorOptions()
                       .dtype(torch::kUInt8)
                       .device(device)
                       .memory_format(c10::MemoryFormat::Contiguous);
  options_copy = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  buffer1 = std::make_shared<torch::Tensor>(torch::zeros(size, options_buffer));
  buffer2 = std::make_shared<torch::Tensor>(torch::zeros(size, options_buffer));
}

void TensorBuffer::set_buffer(uint16_t data[], int numbytes) {
  const auto length = numbytes >> 1;
  const std::lock_guard lock{buffer_lock};
  char *array = (char *)buffer1->data_ptr();
  for (int i = 0; i < length; i = i + 2) {
    // Decode x, y
    const uint16_t y_coord = data[i] & 0x7FFF;
    const uint16_t x_coord = data[i + 1] & 0x7FFF;
    (*(array + shape[1] * x_coord + y_coord))++;
  }
}

void TensorBuffer::set_vector(std::vector<AEDAT::PolarityEvent> events) {
  const std::lock_guard lock{buffer_lock};
  char *array = (char *)buffer1->data_ptr();
  for (auto event : events) {
    (*(array + shape[1] * event.x + event.y))++;
  }
}

at::Tensor TensorBuffer::read() {
  // Swap out old pointer
  {
    const std::lock_guard lock{buffer_lock};
    buffer1.swap(buffer2);
  }
  // Copy and clean
  auto copy = buffer2->to(options_copy, true, true);
  buffer2->index_put_({torch::indexing::Slice()}, false);
  return copy;
}