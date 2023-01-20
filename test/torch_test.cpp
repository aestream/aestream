#include <chrono>
#include <string>

#include <torch/torch.h>

#include <gtest/gtest.h>

#include "generator.hpp"

#include "input/file.hpp"
#include "output/dvs_to_tensor.hpp"

using namespace std::chrono_literals;

TEST(TorchTest, ConvertEvents) {
  const std::atomic<bool> flag = {true};
  auto generator = file_event_generator("example/sample.aedat4", flag);
  auto events = std::vector<AER::Event>();
  int count = 0;
  for (auto event : generator) {
    events.push_back(event);
    count++;
    if (count >= 101)
      break;
  }
  auto tensor = convert_polarity_events(events);

  EXPECT_EQ(torch::_sparse_sum(tensor).item<int>(), count);
  tensor = convert_polarity_events(events, {346, 260});
  EXPECT_EQ(tensor.to_dense().sum().item<int>(), count);
}
TEST(TorchTest, GenerateTensor) {
  const std::atomic<bool> flag = {true};
  auto generator = file_event_generator("example/sample.aedat4", flag);
  auto to_tensor = sparse_tensor_generator(generator, 3s, {346, 260});
  uint64_t sum = 0;
  for (torch::Tensor tensor : to_tensor) {
    sum += torch::_sparse_sum(tensor).item<int>();
  }
  EXPECT_EQ(sum, 117667);
}
TEST(TorchTest, GenerateTensorSmallWindow) {
  const std::atomic<bool> flag = {true};
  auto generator = file_event_generator("example/sample.aedat4", flag);
  auto to_tensor = sparse_tensor_generator(generator, 1ms, {346, 260},
                                           torch::DeviceType::CPU);
  uint64_t sum = 0;
  for (torch::Tensor tensor : to_tensor) {
    sum += torch::_sparse_sum(tensor).item<int>();
  }
  EXPECT_EQ(sum, 117667);
}