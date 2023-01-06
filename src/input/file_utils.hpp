#pragma once

#include <memory>
#include <stdio.h>

void close_file(FILE *fp);

typedef std::unique_ptr<FILE, decltype(&close_file)> unique_file_t;

unique_file_t open_file(const std::string &filename);

template <typename T, typename R, R h(const unique_file_t &),
          T f(const unique_file_t &, const R &, uint64_t *)>
T file_reader(const std::string &filname, const R &, const std::atomic<bool> &);

bool ends_with(std::string const &value, std::string const &ending);

#define HEADER_START 0x25
#define HEADER_END 0x0A
#define BUFFER_SIZE 4096