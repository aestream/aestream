#pragma once

#include <string>
#include <thread>
#include <vector>

#include "../aedat.hpp"

#include <torch/extension.h>
#include <torch/torch.h>

class TensorBuffer {
private:
  const std::vector<int64_t> shape;
  torch::TensorOptions options_buffer;
  torch::TensorOptions options_copy;

  std::mutex buffer_lock;
  std::shared_ptr<torch::Tensor> buffer1;
  std::shared_ptr<torch::Tensor> buffer2;

public:
  TensorBuffer(torch::IntArrayRef size, std::string device);
  void set_buffer(uint16_t data[], int numbytes);
  void set_vector(std::vector<AEDAT::PolarityEvent> events);
  at::Tensor read();
};