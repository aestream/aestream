#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

#include "../aer.hpp"
#include "../generator.hpp"

#include "utils.hpp"

struct DATEvent {
  signed int ts : 32;
  unsigned int x : 14;
  unsigned int y : 14;
  unsigned int p : 4;
} __attribute__((packed));

inline AER::Event dat_decode_event(uint64_t data, size_t overflows);

size_t dat_read_header(const unique_file_t &fp);

std::tuple<AER::Event *, size_t> dat_read_n_events(const unique_file_t &fp, const size_t &n_events);

Generator<AER::Event> dat_stream_events(const unique_file_t &fp);