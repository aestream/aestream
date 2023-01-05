#include "file_dat.hpp"

#include <sys/stat.h>

size_t dat_read_header(const unique_file_t &fp) {
  size_t bytes_read = 0;
  uint8_t c, header_begins;

  do {
    header_begins = 1;
    do {
      bytes_read += fread(&c, 1, 1, fp.get());
      if (header_begins && c != HEADER_START) {
        if (fseek(fp.get(), -1, SEEK_CUR) != 0) {
          throw std::runtime_error("Failed to process .dat file header");
        }
        --bytes_read;
        goto process_length;
      } else {
        header_begins = 0;
      }
    } while (c != HEADER_END);
  } while (1);

  if (bytes_read == 0) {
    throw std::runtime_error("Failed to process .dat file header");
  }

process_length:
  fseek(fp.get(), 0, SEEK_END);
  size_t fileLength = ftell(fp.get());
  fseek(fp.get(), bytes_read + 2, SEEK_SET); // Skip header bytes
  return (fileLength - bytes_read) / sizeof(int64_t);
}

AER::Event *dat_read_all_events(const unique_file_t &fp,
                                const size_t &n_events) {
  uint64_t *buffer = (uint64_t *)malloc(BUFFER_SIZE * sizeof(uint64_t));
  AER::Event *events = (AER::Event *)malloc(n_events * sizeof(AER::Event));
  uint64_t timestep = 0;
  size_t overflows = 0;
  size_t index = 0;
  size_t size = 0;

  do {
    size = fread(buffer, sizeof(*buffer), BUFFER_SIZE, fp.get());

    if (size == 0 && !feof(fp.get())) {
      throw std::runtime_error("Error when processing .dat file");
    }

    for (int i = 0; i < size; i++) {
      AER::Event event = decode_event(buffer[i], overflows);
      if (event.timestamp < timestep) { // Timestep overflow occurred
        overflows++;
        event.timestamp = (overflows << 32) | event.timestamp;
      }
      events[index + i] = event;
    }
    index += size;
  } while (size > 0);
  return events;
}

Generator<AER::Event> dat_read_stream_all_events(const unique_file_t &fp) {
  const auto time_start = std::chrono::high_resolution_clock::now();
  uint64_t buffer[BUFFER_SIZE];
  size_t count = 0;
  uint64_t timestep = 0;
  size_t overflows = 0;
  int n = 0;
  size_t size;
  do {
    size = fread(buffer, sizeof(uint64_t), BUFFER_SIZE, fp.get());

    if (size == 0 && !feof(fp.get())) {
      throw std::runtime_error("Error when processing .dat file");
    }

    for (int i = 0; i < size; i++) {
      AER::Event event = decode_event(buffer[i], overflows);
      if (event.timestamp < timestep) { // Timestep overflow occurred
        overflows++;
        event.timestamp = (overflows << 32) | event.timestamp;
      }
      co_yield event;
    }
  } while (size > 0);
}

Generator<AER::Event> dat_stream_all_events(const unique_file_t &fp,
                                            const std::atomic<bool> &runFlag,
                                            bool ignore_time) {
  const auto time_start = std::chrono::high_resolution_clock::now();
  uint64_t buffer[BUFFER_SIZE];
  size_t count = 0;
  uint64_t timestep = 0;
  size_t overflows = 0;
  int n = 0;
  while (runFlag.load()) {
    const auto size = fread(buffer, sizeof(uint64_t), BUFFER_SIZE, fp.get());

    if (size == 0) {
      if (feof(fp.get())) {
        break;
      } else {
        throw std::runtime_error("Error when processing .dat file");
      }
    }

    for (int i = 0; i < size; i++) {
      AER::Event event = decode_event(buffer[i], overflows);
      if (event.timestamp < timestep) { // Timestep overflow occurred
        overflows++;
        event.timestamp = (overflows << 32) | event.timestamp;
      }

      // Check for time discrepancies
      // if (!ignore_time) {
      //   const int64_t time_diff =
      //       std::chrono::duration_cast<std::chrono::microseconds>(
      //           std::chrono::high_resolution_clock::now() - time_start)
      //           .count();
      //   const int64_t time_offset = timestep - time_diff;
      //   if (time_offset > 5) {
      //     std::this_thread::sleep_for(std::chrono::microseconds(time_offset));
      //   }
      // }
      co_yield event;
    }
  };
}

inline AER::Event decode_event(uint64_t data, size_t overflows) {
  auto event = *(DATEvent *)(&data);
  return AER::Event{event.ts, event.x, event.y, event.p};

  // const uint64_t lower = data & mask_32b;
  // const uint64_t upper = data >> 32;
  // const uint64_t timestep = (overflows << 32) | lower;

  // return AER::Event{timestep, (upper & mask_14b), ((upper >> 14) & mask_14b),
  //                   ((upper >> 28) & mask_4b) > 0};
}