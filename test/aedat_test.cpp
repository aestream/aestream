#include <gtest/gtest.h>
#include "aedat4.hpp"

#include <string>

TEST(FileTest, FailEmptyFile) {
  EXPECT_THROW(AEDAT4("idonotexist.aedat4"), std::runtime_error);
}
TEST(FileTest, ReadAEDATFile) {
  auto file = AEDAT4("../example/davis.aedat4");
  EXPECT_EQ(file.polarity_events.size(), 117667);
}
TEST(FileTest, ReadFileStream) {
  auto generator = file_event_generator("../example/davis.aedat4");
  for (AEDAT::PolarityEvent event : generator) {
    EXPECT_EQ(event.x, 218);
    break;
  }
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}