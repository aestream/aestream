#pragma once

#include <string>
#include <thread>
#include <vector>

#include <torch/extension.h>
#include <torch/torch.h>

#include "../aedat.hpp"

class TensorBuffer {
private:
  const std::vector<int64_t> shape;
  torch::TensorOptions options_buffer;
  torch::TensorOptions options_copy;

  std::mutex buffer_lock;
  std::shared_ptr<torch::Tensor> buffer1;
  std::shared_ptr<torch::Tensor> buffer2;
#ifdef WITH_CUDA
  std::vector<int> offset_buffer;
  int *cuda_device_pointer;
#endif
public:
  TensorBuffer(torch::IntArrayRef size, torch::Device device,
               size_t buffer_size);
  ~TensorBuffer();
  void set_buffer(uint16_t data[], int numbytes);
  void set_vector(std::vector<AEDAT::PolarityEvent> events);
  at::Tensor read();
};