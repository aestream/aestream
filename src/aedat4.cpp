#include <chrono>
#include <string>
#include <thread>

#include "aedat.hpp"
#include "aedat4.hpp"

/**
 * Reads AEDAT events from a files and replays them.
 *
 * @param filename  The path to the file
 * @param ignore_time  Whether to ignore the timestamps and replay the events as
 * fast as possible (true) or enforce that the events are replayed in real-time
 * (false, default).
 * @return A Generator of PolarityEvents
 */
Generator<AEDAT::PolarityEvent>
file_event_generator(const std::string &filename, bool ignore_time) {
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
        // printf("%d, %d\n", event.x, event.y);
        const auto sleep_time = std::min((int64_t)1000, time_offset);
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_time));
      }
    }
    co_yield event;
  }
}