#pragma once

#include <memory>
#include <queue>
#include <string>

#include "../aer.hpp"
#include "../generator.hpp"

#define BUFFER_SIZE 4096

static void close_file(FILE *fp)
{
  if (fp)
  {
    fclose(fp);
  }
}

typedef std::unique_ptr<FILE, decltype(&close_file)> file_t;

// Thanks to https://stackoverflow.com/a/2072890/999865
static bool ends_with(std::string const &value, std::string const &ending)
{
  if (ending.size() > value.size())
    return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

static file_t open_file(const std::string &filename)
{
  file_t fp(fopen(filename.c_str(), "rb"), &close_file);

  if (fp.get() == NULL)
  {
    throw std::invalid_argument("Cannot open file " +
                                filename); // throw std::runtime_error("");
  }
  return fp;
}

static long file_size(FILE *fp) {
  auto origin = ftell(fp);
  fseek(fp, 0, SEEK_END);
  auto size = ftell(fp);
  fseek(fp, origin, SEEK_SET);
  return size;
}

struct FileBase
{
  virtual ~FileBase() = default;
  virtual Generator<AER::Event> stream(const int64_t n_events = -1) = 0;
  virtual std::tuple<std::vector<AER::Event>, size_t>
  read_events(const int64_t n_events = -1) = 0;
};
