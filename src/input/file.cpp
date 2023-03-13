#include "file.hpp"

std::unique_ptr<FileBase> file_base(const std::string &filename) {
  const auto fp = open_file(filename);

  if (ends_with(filename, ".dat")) {
    return std::unique_ptr<FileBase>(new DAT(fp));
  } else if (ends_with(filename, ".aedat4")) {
    return std::unique_ptr<FileBase>(new AEDAT4(fp));
  } else {
    throw std::invalid_argument("Unknown file type " + filename);
  }
}

Generator<AER::Event> file_event_generator(const std::string filename,
                                           const std::atomic<bool> &runFlag) {
  return file_base(filename)->stream(-1);
}
