#include "file_utils.hpp"

void close_file(FILE *fp) { fclose(fp); }

// Thanks to https://stackoverflow.com/a/2072890/999865
bool ends_with(std::string const &value, std::string const &ending) {
  if (ending.size() > value.size())
    return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

unique_file_t open_file(const std::string &filename) {
  unique_file_t fp(fopen(filename.c_str(), "rb"), &close_file);

  if (fp.get() == nullptr) {
    throw std::invalid_argument("Cannot open file " + filename);
  }
  return fp;
}

// template <typename T, typename R, R h(const unique_file_t &),
//           T f(const unique_file_t &, const R &, uint64_t *)>
// T file_reader(const std::string &filename, const std::atomic<bool> &runFlag) {
//   const auto fp = open_file(filename);
//   const R header = h(fp);
//   const uint64_t buffer[BUFFER_SIZE];

//   const auto out = f(fp, header, runFlag);
//   // delete buffer[];
//   return out;
// }