#include <string>

#include <gtest/gtest.h>

#include "../src/aedat4.hpp"
#include "../src/input/file.hpp"

TEST(FileTest, FailEmptyFile) {
  EXPECT_THROW(AEDAT4("idonotexist.aedat4"), std::runtime_error);
}
TEST(FileTest, ReadAEDATFile) {
  auto file = AEDAT4("../example/davis.aedat4");
  const size_t expected = 117667;
  ASSERT_EQ(file.polarity_events.size(), expected);
}
TEST(FileTest, ReadFileStream) {
  auto generator = file_event_generator("../example/davis.aedat4", false);
  for (auto event : generator) {
    EXPECT_EQ(event.x, 218);
    break;
  }
}
