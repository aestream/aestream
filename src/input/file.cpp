#include "file.hpp"

// Thanks to https://stackoverflow.com/a/2072890/999865
inline bool ends_with(std::string const &value, std::string const &ending) {
  if (ending.size() > value.size())
    return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

void close_file(FILE *fp) { fclose(fp); }

Generator<AEDAT::PolarityEvent>
file_event_generator(const std::string filename,
                     const std::atomic<bool> &runFlag, bool ignore_time) {
  unique_file_t fp(fopen(filename.c_str(), "rb"), &close_file);

  if (fp.get() == nullptr) {
    throw std::invalid_argument("Cannot open file " + filename);
  }

  if (ends_with(filename, "dat")) {
    const auto jump = jump_header(fp);
    if (jump == 0) {
      throw std::runtime_error("Failed to process .dat file header");
    } else {
      fseek(fp.get(), 2, SEEK_CUR); // Skip header bytes
    }

    std::vector<uint64_t> buffer(BUFFER_SIZE);
    size_t count = 0;
    uint64_t timestep = 0;
    size_t overflows = 0;
    int n = 0;
    while (runFlag.load()) {
      const auto size =
          fread(buffer.data(), sizeof(buffer.front()), buffer.size(), fp.get());

      if (size == 0) {
        if (feof(fp.get())) {
          break;
        } else {
          throw std::runtime_error("Error when processing .dat file");
        }
      }

      for (int i = 0; i < size; i++) {
        DATEvent event = *(DATEvent *)(&buffer[i]);
        const auto newTimestep = le32toh(event.ts);
        if (newTimestep < timestep) { // Timestep overflow occurred
          overflows++;
          timestep = (overflows << 32) | newTimestep;
        } else {
          timestep = newTimestep;
        }
        co_yield AEDAT::PolarityEvent{timestep, le16toh(event.x),
                                      le16toh(event.y), true, event.p > 0};
      }
    };

  } else if (ends_with(filename, "aedat4")) {
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
  } else {
    throw std::invalid_argument("Unknown file ending for file " + filename);
  }
}

size_t jump_header(const unique_file_t &fp_in) {
  size_t bytes_read = 0;
  uint8_t c, header_begins;
  do {
    header_begins = 1;
    do {
      bytes_read += fread(&c, 1, 1, fp_in.get());
      if (header_begins && c != HEADER_START) {
        if (fseek(fp_in.get(), -1, SEEK_CUR) != 0) {
          throw std::runtime_error("Failed to process .dat file header");
        }
        return --bytes_read;
      } else {
        header_begins = 0;
      }
    } while (c != HEADER_END);
  } while (1);
  return 0;
}