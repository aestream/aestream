#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

#include "../aer.hpp"
#include "../generator.hpp"

#include "utils.hpp"

struct DAT : FileBase {

  struct DATEvent {
    signed int ts : 32;
    unsigned int x : 14;
    unsigned int y : 14;
    unsigned int p : 4;
  } __attribute__((packed));

  Generator<AER::Event> stream(const int64_t n_events = -1) {
    static const size_t STREAM_BUFFER_SIZE = 4096;
    uint64_t buffer[STREAM_BUFFER_SIZE];
    uint64_t timestep = 0;
    size_t overflows = 0;
    size_t size;
    do {
      size = fread(buffer, sizeof(uint64_t), STREAM_BUFFER_SIZE, fp.get());

      if (size == 0 && !feof(fp.get())) {
        throw std::runtime_error("Error when processing .dat file");
      }

      for (int i = 0; i < size; ++i) {
        AER::Event event = dat_decode_event(buffer[i], overflows);
        if (event.timestamp < timestep) { // Timestep overflow occurred
          overflows++;
          event.timestamp = (overflows << 32) | event.timestamp;
        }
        co_yield event;
      }
    } while (size > 0);
  }

  std::tuple<std::vector<AER::Event>, size_t>
  read_events(const int64_t n_events = -1) {
    static const size_t READ_BUFFER_SIZE = 4096;
    const size_t buffer_size = n_events > 0 ? n_events : READ_BUFFER_SIZE;
    const size_t event_array_size =
        n_events > 0 ? n_events : total_number_of_events;

    std::vector<uint64_t> buffer_vector = std::vector<uint64_t>();
    buffer_vector.resize(buffer_size);
    uint64_t *buffer = buffer_vector.data();
    std::vector<AER::Event> events{};
    events.reserve(event_array_size);
    size_t timestep = 0, overflows = 0, index = 0, size = 0;
    do {
      size = fread(buffer, sizeof(uint64_t), buffer_size, fp.get());

      if (size == 0 && !feof(fp.get())) {
        throw std::runtime_error("Error when processing .dat file");
      }

      if (size > n_events - index) {
        fseek(fp.get(), size - (n_events - index), SEEK_CUR); // Re-align file
        size = n_events - index;
      }

      for (size_t i = 0; i < size; ++i) {
        AER::Event event = dat_decode_event(buffer[i], overflows);
        if (event.timestamp < timestep) { // Timestep overflow occurred
          overflows++;
          event.timestamp = (overflows << 32) | event.timestamp;
        }
        events.push_back(event);
      }
      index += size;
    } while (size > 0 && n_events - index > 0);
    return {events, index};
  }

  explicit DAT(const std::string &filename) : DAT(open_file(filename)) {}
  explicit DAT(file_t &&fp)
      : fp(std::move(fp)), total_number_of_events{dat_read_header()} {}

private:
  const file_t fp;
  const size_t total_number_of_events;

  static constexpr char HEADER_END = 0x0A;   // \n
  static constexpr char HEADER_START = 0x25; // %

  size_t dat_read_header() {
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
          return process_length(bytes_read);
        } else {
          header_begins = 0;
        }
      } while (c != HEADER_END);
    } while (1);

    if (bytes_read == 0) {
      throw std::runtime_error("Failed to process .dat file header");
    }

    return process_length(bytes_read);
  }

  size_t process_length(size_t bytes_read) {
    fseek(fp.get(), 0, SEEK_END);
    size_t file_length = ftell(fp.get());
    fseek(fp.get(), bytes_read + 2, SEEK_SET); // Skip header bytes
    return (file_length - bytes_read) / sizeof(int64_t);
  }

  inline AER::Event dat_decode_event(uint64_t data, size_t overflows) {
    auto event = *(DATEvent *)(&data);
    return AER::Event{
        static_cast<uint64_t>(event.ts), static_cast<uint16_t>(event.x),
        static_cast<uint16_t>(event.y), static_cast<bool>(event.p)};
  }
};