#include <chrono>
#include <string>
#include <thread>

#include "../aedat4.hpp"
#include "../generator.hpp"
#include "file.hpp"

#include <iostream>

Generator<AEDAT::PolarityEvent> file_event_generator(const std::string filename,
                                                     bool ignore_time) {
  AEDAT4 aedat_file = AEDAT4(filename);
  const auto polarity_events = aedat_file.polarity_events;
  const auto time_start = std::chrono::high_resolution_clock::now();
  const int64_t time_start_us = polarity_events[0].timestamp;

  for (auto event : polarity_events) {
    // Sleep to align with real-time, unless ignore_time is set
    if (!ignore_time) {
      const int64_t time_diff =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::high_resolution_clock::now() - time_start)
              .count();
      const int64_t file_diff = event.timestamp - time_start_us;
      const int64_t time_offset = file_diff - time_diff;
      if (time_offset > 1000) {
        const auto sleep_time = std::min((int64_t)1000, time_offset);
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_time));
      }
    }
    co_yield event;
  }
}