#pragma once

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <string>
#include <thread>

#include "../aer.hpp"
#include "../generator.hpp"

#include "../file/aedat4.hpp"
#include "../file/dat.hpp"
#include "../file/utils.hpp"

/**
 * Attempts to open a file containing address-event representations.
 *
 * @param filename The path to the file
 * @return A FileBase pointer
 * @throws std::invalid_argument if the file could not be found or opened
 */
std::unique_ptr<FileBase> file_base(const std::string &filename);

/**
 * Reads AEDAT events from a file and replays them either in real-time
 * (ignore_time = false) or as fast as possible (ignore_time = true).
 *
 * @param filename  The path to the file
 * @param run_flag  A flag to stop the reading
 * @param ignore_time  Whether to ignore the timestamps and replay the events as
 * fast as possible (true) or enforce that the events are replayed in real-time
 * (false, default).
 * @return A Generator of Events
 */
Generator<AER::Event> file_event_generator(const std::string filename,
                                           const std::atomic<bool> &run_flag);