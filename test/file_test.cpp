#include <string>

#include <gtest/gtest.h>

#include "input/file.hpp"

TEST(FileTest, FailEmptyFile) {
  EXPECT_THROW(open_event_file("idonotexist.aedat4"), std::invalid_argument);
  EXPECT_THROW(open_event_file("idonotexist.dat"), std::invalid_argument);
  EXPECT_THROW(open_event_file("idonotexist.csv"), std::invalid_argument);
}

TEST(FileTest, ReadCSVFile) {
  auto file = open_event_file("example/sample.csv");
  auto [events, size] = file->read_events(-1);
  const size_t expected = 100;
  ASSERT_EQ(size, expected);
  ASSERT_EQ(events[99].timestamp, 99);
}
TEST(FileTest, ReadDATFile) {
  auto file = open_event_file("example/sample.dat");
  auto [events, size] = file->read_events(-1);
  const size_t expected = 539481;
  ASSERT_EQ(size, expected);
  ASSERT_EQ(events[0].timestamp, 0);
  ASSERT_EQ(events[0].x, 237);
  ASSERT_EQ(events[0].y, 121);
}
TEST(FileTest, ReadDATFilePart) {
  auto file = open_event_file("example/sample.dat");
  auto [events, size] = file->read_events(10000);
  const size_t expected = 10000;
  ASSERT_EQ(size, expected);
}
TEST(FileTest, ReadDATFileParts) {
  auto file = open_event_file("example/sample.dat");
  auto [events1, size1] = file->read_events(10000);
  const size_t expected1 = 10000;
  ASSERT_EQ(size1, expected1);
  auto [events2, size2] = file->read_events(-1);
  const size_t expected2 = 539481 - 10000;
  ASSERT_EQ(size2, expected2);
}
TEST(FileTest, ReadAEDAT4File) {
  auto file = open_event_file("example/sample.aedat4");
  auto [events, size] = file->read_events(-1);
  const size_t expected = 117667;
  ASSERT_EQ(size, expected);
  ASSERT_EQ(events[0].timestamp, 1633953690975950);
  ASSERT_EQ(events[0].x, 218);
  ASSERT_EQ(events[0].y, 15);
}
TEST(FileTest, ReadAEDAT4FilePart) {
  auto file = open_event_file("example/sample.aedat4");
  auto [events, size] = file->read_events(10000);
  const size_t expected = 10000;
  ASSERT_EQ(size, expected);
}
TEST(FileTest, ReadAEDAT4FileParts) {
  auto file = open_event_file("example/sample.aedat4");
  auto [events1, size1] = file->read_events(10000);
  const size_t expected1 = 10000;
  ASSERT_EQ(size1, expected1);
  auto [events2, size2] = file->read_events(-1);
  const size_t expected2 = 117667 - 10000;
  ASSERT_EQ(size2, expected2);
}
TEST(FileTest, StreamAEDAT4File) {
  auto handle = open_event_file("example/sample.aedat4");
  auto generator = handle->stream();
  const size_t expected = 117667;
  int count = 0;
  for (auto event : generator) {
    count++;
  }
  ASSERT_EQ(count, expected);
}
TEST(FileTest, StreamDATFile) {
  auto handle = open_event_file("example/sample.dat");
  auto generator = handle->stream();
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
  auto handle = open_event_file("example/sample.aedat4");
  auto generator = handle->stream();
  for (auto event : generator) {
    EXPECT_EQ(event.x, 218);
    break;
  }
}