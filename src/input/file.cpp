#include "file.hpp"

std::unique_ptr<FileBase> open_event_file(const std::string &filename) {
  auto fp = open_file(filename);

  if (ends_with(filename, ".dat")) {
    return std::unique_ptr<FileBase>(new DAT(std::move(fp)));
  } else if (ends_with(filename, ".aedat4")) {
    return std::unique_ptr<FileBase>(new AEDAT4(std::move(fp)));
  } else {
    throw std::invalid_argument("Unknown file type " + filename);
  }
}
