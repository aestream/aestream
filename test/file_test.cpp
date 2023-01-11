#include <string>

#include <gtest/gtest.h>

#include "input/file.hpp"

TEST(FileTest, FailEmptyFile) {
  const std::atomic<bool> flag = {true};
  EXPECT_THROW(file_event_generator("idonotexist.aedat4", flag).begin(),
               std::invalid_argument);
  EXPECT_THROW(file_event_generator("idonotexist.dat", flag).begin(),
               std::invalid_argument);
}
TEST(FileTest, ReadAEDATFile) {
  auto file = AEDAT4("example/sample.aedat4");
  const size_t expected = 117667;
  ASSERT_EQ(file.polarity_events.size(), expected);
}
TEST(FileTest, StreamAEDATFile) {
  const std::atomic<bool> flag = {true};
  auto generator = file_event_generator("example/sample.aedat4", flag);
  const size_t expected = 117667;
  int count = 0;
  for (auto event : generator) {
    count++;
  }
  ASSERT_EQ(count, expected);
}
TEST(FileTest, StreamDATFile) {
  const std::atomic<bool> flag = {true};
  auto generator = file_event_generator("example/sample.dat", flag);
  int i = 0;
  int x;
  size_t count = 0;
  for (auto event : generator) {
    x = event.x;
    count++;
  }
  EXPECT_EQ(x, 210);
  EXPECT_EQ(count, 539481);
}
TEST(FileTest, ReadFileStream) {
  auto generator = file_event_generator("example/sample.aedat4", false);
  for (auto event : generator) {
    EXPECT_EQ(event.x, 218);
    break;
  }
}
