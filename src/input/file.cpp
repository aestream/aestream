#include "file.hpp"

Generator<AER::Event> file_event_generator(const std::string filename,
                                           const std::atomic<bool> &runFlag) {
  auto fp = open_file(filename);

  if (ends_with(filename, ".dat")) {
    return dat_stream_events(filename);
  } else if (ends_with(filename, ".aedat4")) {
    return AEDAT4(fp).stream();
  } else {
    throw std::invalid_argument("Unknown file type " + filename);
  }
}
