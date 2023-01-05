#pragma once

#include <atomic>
#include <chrono>
#include <thread>

#include "../aedat.hpp"
#include "../aer.hpp"
#include "../generator.hpp"

#include "file_utils.hpp"

struct DATEvent {
  signed int ts : 32;
  unsigned int x : 14;
  unsigned int y : 14;
  unsigned int p : 4;
} __attribute__((packed));

static const uint64_t mask_4b = 0xFU, mask_14b = 0x3FFFU,
                      mask_32b = 0xFFFFFFFFU;

size_t dat_read_header(const unique_file_t &fp);

AER::Event *dat_read_all_events(const unique_file_t &fp,
                                const size_t &n_events);

Generator<AER::Event> dat_read_stream_all_events(const unique_file_t &fp);

Generator<AER::Event> dat_stream_all_events(const unique_file_t &fp,
                                            const std::atomic<bool> &runFlag,
                                            bool ignore_time = false);

inline AER::Event decode_event(uint64_t data, size_t overflows);