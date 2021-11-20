#include <chrono>
#include <torch/torch.h>

#include <gtest/gtest.h>

#include "aedat4.hpp"
#include "dvsToTensor.hpp"
#include "generator.hpp"

using namespace std::chrono_literals;

TEST(TorchTest, GenerateTensor) {
  auto generator = file_event_generator("../example/davis.aedat4");
  auto to_tensor =
      dense_tensor_generator(generator, std::chrono::microseconds(1s));
  uint64_t sum = 0;
  for (torch::Tensor tensor : to_tensor) {
    sum += tensor.sum().item<int>();
  }
  EXPECT_EQ(sum, 117667);
}
// EXPECT_EQ(file.polarity_events.size(), 117667);