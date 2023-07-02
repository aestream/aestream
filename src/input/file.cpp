#include "file.hpp"

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <string>
#include <thread>

#include "../aer.hpp"
#include "../generator.hpp"

#include "../file/aedat4.hpp"
#include "../file/csv.hpp"
#include "../file/dat.hpp"
#include "../file/evt3.hpp"
#include "../file/utils.hpp"

std::unique_ptr<FileBase> open_event_file(const std::string &filename) {
  auto fp = open_file(filename);

  if (ends_with(filename, ".dat")) {
    return std::unique_ptr<FileBase>(new DAT(std::move(fp)));
  } else if (ends_with(filename, ".aedat4")) {
    return std::unique_ptr<FileBase>(new AEDAT4(std::move(fp)));
  } else if (ends_with(filename,
                       ".raw")) { // Note the .raw file ending for EVT3
    return std::unique_ptr<FileBase>(new EVT3(std::move(fp)));
  } else if (ends_with(filename, ".csv")) {
    return std::unique_ptr<FileBase>(new CSV(filename));
  } else {
    throw std::invalid_argument("Unknown file type " + filename);
  }
}
