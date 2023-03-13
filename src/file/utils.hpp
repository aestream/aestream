#pragma once

#include <memory>
#include <queue>
#include <string>

#include "../aer.hpp"
#include "../generator.hpp"

struct FileBase {
  virtual ~FileBase() = default;
  virtual Generator<AER::Event> stream(size_t n_events = -1) = 0;
  virtual std::tuple<AER::Event *, size_t>
  read_events(const size_t &n_events = -1) = 0;
};

void close_file(FILE *fp);

// typedef std::unique_ptr<FILE, decltype(&close_file)> unique_file_t;
typedef std::unique_ptr<FILE, decltype(&close_file)> file_t;

file_t open_file(const std::string &filename);

bool ends_with(std::string const &value, std::string const &ending);

#define HEADER_START 0x25
#define HEADER_END 0x0A
#define BUFFER_SIZE 4096
