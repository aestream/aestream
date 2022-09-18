#include "tensor_buffer.hpp"

// CUDA functions
uint32_t *alloc_memory_cuda(size_t buffer_size);
void free_memory_cuda(uint32_t *cuda_device_pointer);
void index_increment_cuda(torch::Tensor array, std::vector<uint32_t> events,
                          uint32_t *event_device_pointer);

// TensorBuffer constructor
TensorBuffer::TensorBuffer(torch::IntArrayRef size, torch::Device device,
                           size_t buffer_size)
    : shape(size.vec()) {
  options_buffer = torch::TensorOptions()
                       .dtype(torch::kUInt8)
                       .device(device)
                       .memory_format(c10::MemoryFormat::Contiguous);
  options_copy = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  buffer1 = std::make_shared<torch::Tensor>(torch::zeros(size, options_buffer));
  buffer2 = std::make_shared<torch::Tensor>(torch::zeros(size, options_buffer));
  if (device == torch::DeviceType::CUDA) {
    cuda_device_pointer = alloc_memory_cuda(buffer_size);
  }
}

TensorBuffer::~TensorBuffer() {
  if (options_buffer.device() == torch::DeviceType::CUDA) {
    free_memory_cuda(cuda_device_pointer);
  }
}

void TensorBuffer::set_buffer(uint16_t data[], int numbytes) {
  const auto length = numbytes >> 1;
  const std::lock_guard lock{buffer_lock};
  uint8_t *array = buffer1->data_ptr<uint8_t>();
  for (int i = 0; i < length; i = i + 2) {
    // Decode x, y
    const uint16_t y_coord = data[i] & 0x7FFF;
    const uint16_t x_coord = data[i + 1] & 0x7FFF;
    (*(array + shape[1] * x_coord + y_coord))++;
  }
}

void TensorBuffer::set_vector(std::vector<AEDAT::PolarityEvent> events) {
  const std::lock_guard lock{buffer_lock};
  if (options_buffer.device() == torch::DeviceType::CUDA) {
    std::vector<uint32_t> offsets;
    for (auto event : events) {
      offsets.push_back(shape[1] * event.x + event.y);
    }
    index_increment_cuda(*buffer1, offsets, cuda_device_pointer);
  } else {
    auto *array = buffer1->data_ptr<uint8_t>();
    for (auto event : events) {
      (*(array + shape[1] * event.x + event.y))++;
    }
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
  buffer2->index_put_({torch::indexing::Slice()}, 0);
  return copy;
}