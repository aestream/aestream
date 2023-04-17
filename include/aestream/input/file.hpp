#pragma once

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <string>
#include <thread>

#include "aestream/aer.hpp"
#include "aestream/generator.hpp"

#include "aestream/file/aedat4.hpp"
#include "aestream/file/dat.hpp"
#include "aestream/file/utils.hpp"

/**
 * Attempts to open a file containing address-event representations.
 *
 * @param filename The path to the file
 * @return A FileBase pointer
 * @throws std::invalid_argument if the file could not be found or opened
 */
std::unique_ptr<FileBase> open_event_file(const std::string &filename);