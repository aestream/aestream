#include <string>

#include <gtest/gtest.h>

#include "aer.hpp"
#include "generator.hpp"
#include "process/transformation.hpp"

Generator<AER::Event> generate(size_t n) {
  for (int i = 0; i < n; ++i) {
    co_yield AER::Event{i, i, i, 0};
  }
}

TEST(ProcessTest, TransformEventTime) {
  auto generator = generate(100);
  auto filtered = transformation_event_generator(generator, "", trans::no_trans, 50, 50, 2, 2);
  size_t sum = 0;
  for (auto event : filtered) {
    EXPECT_EQ(event.timestamp % 2, 0);
    sum++;
  }
  EXPECT_EQ(sum, 50);
}

TEST(ProcessTest, TransformEventSpace) {
  auto generator = generate(100);
  auto filtered = transformation_event_generator(generator, "", trans::no_trans, 50, 50, 2, 2);
  for (auto event : filtered) {
    EXPECT_EQ(event.x % 2, 0); // Spatial sampling
    EXPECT_EQ(event.y % 2, 0);
    EXPECT_TRUE(event.x <= 100); // Spatial scaling
    EXPECT_TRUE(event.y <= 100);
  }
}
