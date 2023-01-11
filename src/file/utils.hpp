#pragma once

#include <memory>
#include <queue>
#include <string>

void close_file(FILE *fp);

// typedef std::unique_ptr<FILE, decltype(&close_file)> unique_file_t;
typedef std::shared_ptr<FILE> shared_file_t;

shared_file_t open_file(const std::string &filename);

template <typename T, typename H, H h(const shared_file_t &),
          T f(const shared_file_t &, const H &, uint64_t *)>
T file_reader(const std::string &filename, const H &header);

bool ends_with(std::string const &value, std::string const &ending);

// template <typename T, typename R>
// std::queue<T> file_to_parts(const unique_file_t &fp, const size_t &size,
//                             const R &header);

// template <typename T, typename R>
// std::queue<T> file_to_intervals(const unique_file_t &fp, const size_t
// &interval,
//                                 const R &header);

#define HEADER_START 0x25
#define HEADER_END 0x0A
#define BUFFER_SIZE 4096
