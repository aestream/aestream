#include "tensor_buffer.hpp"

namespace nb = nanobind;

template <typename scalar_t>
// inline std::unique_ptr<scalar_t[], void (*)(scalar_t *)>
inline std::unique_ptr<scalar_t[], void (*)(scalar_t *)>
allocate_buffer(const size_t &length, std::string device) {
#ifdef USE_CUDA
  if (device == "cuda") {
    std::unique_ptr<scalar_t[], void (*)(scalar_t *)> buffer_ptr(
        static_cast<scalar_t *>(alloc_memory_cuda(length, sizeof(scalar_t))),
        &delete_cuda_buffer<scalar_t>);
    return buffer_ptr;
  }
#endif
  return std::unique_ptr<scalar_t[], void (*)(scalar_t *)>(
      new scalar_t[length]{0}, delete_cpu_buffer<scalar_t>);
}

// TensorBuffer constructor
TensorBuffer::TensorBuffer(std::vector<size_t> size, std::string device,
                           size_t buffer_size)
    : shape(size), device(device) {
#ifdef USE_CUDA // Initialize CUDA buffers
  if (device == "cuda") {
    cuda_buffer = allocate_buffer<int>(buffer_size, device);
    offset_buffer = std::vector<int>(buffer_size);
  }
#endif
  // If device is GeNN, allocate suitably sized bitmask
  if (device == "genn") {
    if (shape.size() == 3) {
      const size_t bitmask_words = ((shape[0] * shape[1] * shape[2]) + 31) / 32;
      genn_events.resize(bitmask_words, 0);
    } else if (shape.size() == 2) {
      const size_t bitmask_words = ((shape[0] * shape[1]) + 31) / 32;
      genn_events.resize(bitmask_words, 0);
    } else {
      throw std::runtime_error("Unsupported shape");
    }
  } else {
    buffer1 = allocate_buffer<float>(size[0] * size[1], device);
    buffer2 = allocate_buffer<float>(size[0] * size[1], device);
  }
}

void TensorBuffer::set_buffer(uint16_t data[], int numbytes) {
  const auto length = numbytes >> 1;
  const std::lock_guard lock{buffer_lock};
#ifdef USE_CUDA
  if (device == "cuda") {
    offset_buffer.clear();
    offset_buffer.reserve(length);
    for (int i = 0; i < length; i = i + 2) {
      // Decode x, y
      const uint16_t y_coord = data[i] & 0x7FFF;
      const uint16_t x_coord = data[i + 1] & 0x7FFF;
      offset_buffer.push_back(shape[1] * x_coord + y_coord);
    }
    index_increment_cuda(buffer1.get(), offset_buffer.data(),
                         offset_buffer.size(), cuda_buffer.get());
    return;
  }
#endif
  if (device == "genn") {
    // Loop through events
    for (int i = 0; i < length; i = i + 2) {
      // Decode x, y coordinates and set event in GeNN format
      const int y_coord = data[i] & 0x7FFF;
      const int x_coord = data[i + 1] & 0x7FFF;
      set_genn_event(x_coord, y_coord, true);
    }
  } else {
    for (int i = 0; i < length; i = i + 2) {
      // Decode x, y
      const int16_t y_coord = data[i] & 0x7FFF;
      const int16_t x_coord = data[i + 1] & 0x7FFF;
      assign_event(buffer1.get(), x_coord, y_coord);
    }
  }
}

void TensorBuffer::set_vector(std::vector<AER::Event> &events) {
  const std::lock_guard lock{buffer_lock};
#ifdef USE_CUDA
  if (device == "cuda") {
    offset_buffer.clear();
    offset_buffer.reserve(events.size());
    for (size_t i = 0; i < events.size(); i++) {
      offset_buffer.push_back(shape[1] * events[i].x + events[i].y);
    }
    index_increment_cuda(buffer1.get(), offset_buffer.data(),
                         offset_buffer.size(), cuda_buffer.get());
    return;
  }
#endif
  if (device == "genn") {
    // Loop through events
    for (const auto &event : events) {
      set_genn_event(event.x, event.y, event.polarity);
    }
  } else {
    for (auto event : events) {
      assign_event(buffer1.get(), event.x, event.y);
    }
  }
}

template <typename R>
inline void TensorBuffer::assign_event(R *array, int16_t x, int16_t y) {
  (*(array + shape[1] * x + y))++;
}

std::unique_ptr<BufferPointer> TensorBuffer::read() {
  // Swap out old pointer
  {
    const std::lock_guard lock{buffer_lock};
    buffer1.swap(buffer2);
  }
  // Create new buffer and swap with old
  buffer_t buffer3 = allocate_buffer<float>(shape[0] * shape[1], device);

  buffer2.swap(buffer3);
  // Return pointer
  return std::unique_ptr<BufferPointer>(
      new BufferPointer(std::move(buffer3), shape, device));
}

void TensorBuffer::read_genn(uint32_t *bitmask, size_t size) {
  // Check size
  assert(size == genn_events.size());

  // Lock
  // **THINK** we COULD double-buffer but I suspect not worth it
  std::lock_guard lock{buffer_lock};

  // Copy bitmask to GeNN-owned pointer
  std::copy(genn_events.cbegin(), genn_events.cend(), bitmask);

  // Zero bitmask
  std::fill(genn_events.begin(), genn_events.end(), 0);
}

#include <iostream>

BufferPointer::BufferPointer(buffer_t &&data, const std::vector<size_t> &shape,
                             const std::string &device)
    : data(std::move(data)), shape(shape), device(device) {}

template <typename tensor_type>
inline tensor_type BufferPointer::to_tensor_type() {
  const size_t s[2] = {shape[0], shape[1]};
  float *ptr = data.release();
  int32_t device_type =
      device == "cuda" ? nb::device::cuda::value : nb::device::cpu::value;
  nb::capsule owner;
  #ifdef USE_CUDA
    if (device == "cuda") {
      owner = nb::capsule(ptr, [](void *p) noexcept { free_memory_cuda((float *)p); });
    } else {
  #endif
      owner = nb::capsule(ptr, [](void *p) noexcept { delete[] (float *) p; });
  #ifdef USE_CUDA
    }
  #endif
  return tensor_type(ptr, 2, s, owner, /* owner */
                     nullptr,          /* strides */
                     nanobind::dtype<float>(), device_type);
}

tensor_numpy BufferPointer::to_numpy() {
  return to_tensor_type<tensor_numpy>();
}

tensor_jax BufferPointer::to_jax() { return to_tensor_type<tensor_jax>(); }

tensor_torch BufferPointer::to_torch() {
  return to_tensor_type<tensor_torch>();
}
