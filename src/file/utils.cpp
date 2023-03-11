#include "utils.hpp"

void close_file(FILE *fp) {
  if (fp) {
    fclose(fp);
  }
}

// Thanks to https://stackoverflow.com/a/2072890/999865
bool ends_with(std::string const &value, std::string const &ending) {
  if (ending.size() > value.size())
    return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

#include <iostream>

shared_file_t open_file(const std::string &filename) {
  shared_file_t fp(fopen(filename.c_str(), "rb"), &close_file);

  if (fp.get() == NULL) {
    throw std::invalid_argument("Cannot open file " +
                                filename); // throw std::runtime_error("");
  }
  return fp;
}

template <typename T, typename H, H h(const shared_file_t &),
          T f(const shared_file_t &, const H &, uint64_t *)>
T file_reader(const std::string &filename) {
  const auto fp = open_file(filename);
  const H header = h(fp);
  uint64_t buffer[BUFFER_SIZE];

  const auto out = f(fp, header, buffer);
  // delete[] buffer;
  return out;
}
