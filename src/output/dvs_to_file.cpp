
#include <fstream>
#include <string>

#include "aedat.hpp"
#include "aedat4.hpp"
#include "generator.hpp"
#include "dvs_to_file.hpp"

void dvs_to_file_aedat(Generator<AEDAT::PolarityEvent> &input_generator,
                       const std::string &filename, size_t bufferSize) {
  std::fstream fileOutput;
  fileOutput.open(filename, std::fstream::in | std::fstream::out |
                                std::fstream::binary | std::fstream::trunc);

  // Header
  auto headerOffset = AEDAT4::save_header(fileOutput);

  // Events
  std::vector<AEDAT::PolarityEvent> events;
  size_t sum = 0;
  uint64_t timeStart = 0;
  uint64_t timeEnd = 0;
  for (auto event : input_generator) {
    events.push_back(event);

    if (events.size() >= bufferSize) {
      AEDAT4::save_events(fileOutput, events);
      events.clear();
      timeEnd = events.back().timestamp;
    }

    if (timeStart == 0) {
      timeStart = event.timestamp;
    }
    sum++;
  }
  if (events.size() > 0) {
    AEDAT4::save_events(fileOutput, events);
    timeEnd = events.back().timestamp;
  }

  // Footer
  AEDAT4::save_footer(fileOutput, headerOffset, timeStart, timeEnd, sum);
  fileOutput.flush();
  fileOutput.close();
}

void dvs_to_file_txt(Generator<AEDAT::PolarityEvent> &input_generator,
                     const std::string &filename) {
  std::fstream fileOutput;
  fileOutput.open(filename, std::fstream::app);

  for (AEDAT::PolarityEvent event : input_generator) {
    if (event.valid == true) {
      fileOutput << "DVS " << event.timestamp << " " << event.x << " "
                 << event.y << " " << event.polarity << std::endl;
    }
  }

  fileOutput.close();
}