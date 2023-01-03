#pragma once

#include <atomic>
#include <chrono>
#include <endian.h>
#include <exception>
#include <memory>
#include <string>
#include <thread>

#include <stdio.h>

#include "../aedat.hpp"
#include "../aedat4.hpp"
#include "../generator.hpp"

/**
 * Reads AEDAT events from a file and replays them either in real-time
 * (ignore_time = false) or as fast as possible (ignore_time = true).
 *
 * @param filename  The path to the file
 * @param run_flag  A flag to stop the reading
 * @param ignore_time  Whether to ignore the timestamps and replay the events as
 * fast as possible (true) or enforce that the events are replayed in real-time
 * (false, default).
 * @return A Generator of PolarityEvents
 */
Generator<AEDAT::PolarityEvent>
file_event_generator(const std::string filename,
                     const std::atomic<bool> &run_flag,
                     bool ignore_time = false);

struct DATEvent {
  signed int ts : 32;
  unsigned int x : 14;
  unsigned int y : 14;
  unsigned int p : 4;
} __attribute__((packed));

void close_file(FILE *fp);

using unique_file_t = std::unique_ptr<FILE, decltype(&close_file)>;

inline bool ends_with(std::string const &value, std::string const &ending);

#define HEADER_START 0x25
#define HEADER_END 0x0A
#define BUFFER_SIZE 128
size_t jump_header(const unique_file_t &fp_in);