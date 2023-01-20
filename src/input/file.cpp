#include "file.hpp"

Generator<AER::Event> file_event_generator(const std::string filename,
                                           const std::atomic<bool> &runFlag) {
  if (ends_with(filename, ".dat")) {
    return dat_stream_events(filename);
  } else if (ends_with(filename, ".aedat4")) {
    return AEDAT4::aedat_to_stream(filename);
  } else {
    throw std::invalid_argument("Unknown file ending for file " + filename);
  }
}
