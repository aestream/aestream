#include "file.hpp"

Generator<AER::Event> file_event_generator_aedat(const std::string &filename,
                                           const std::atomic<bool> &runFlag,
                                           bool ignore_time) {
  AEDAT4 aedat_file = AEDAT4(filename);
  const auto polarity_events = aedat_file.polarity_events;
  const auto time_start = std::chrono::high_resolution_clock::now();
  const int64_t time_start_us = polarity_events[0].timestamp;

  for (auto event : polarity_events) {
    if (!runFlag.load()) { // Stop if requested
      break;
    }
    // Sleep to align with real-time, unless ignore_time is set
    if (!ignore_time) {
      const int64_t time_diff =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::high_resolution_clock::now() - time_start)
              .count();
      const int64_t file_diff = event.timestamp - time_start_us;
      const int64_t time_offset = file_diff - time_diff;
      if (time_offset > 1000) {
        std::this_thread::sleep_for(std::chrono::microseconds(time_offset));
      }
    }
    co_yield event;
  }
}

Generator<AER::Event> file_event_generator(const std::string filename,
                                           const std::atomic<bool> &runFlag,
                                           bool ignore_time) {

  if (ends_with(filename, ".dat")) {
    const unique_file_t &fp = open_file(filename);
    const auto n_events = dat_read_header(fp);
    return dat_stream_events(fp);
  } else if (ends_with(filename, ".aedat4")) {
    return file_event_generator_aedat(filename, runFlag, ignore_time);
  } else {
    throw std::invalid_argument("Unknown file ending for file " + filename);
  }
}
