
#include <fstream>
#include <string>

#include "../aedat.hpp"
#include "../generator.hpp"
#include "dvs_to_file.hpp"

void dvs_to_file(Generator<AEDAT::PolarityEvent> &input_generator,
                 const std::string &filename) {
  std::fstream fileOutput;
  fileOutput.open(filename, std::fstream::app);

  for (AEDAT::PolarityEvent event : input_generator) {
    if (event.valid == true) {
      fileOutput << "DVS " << event.timestamp << " " << event.x << " "
                 << event.y << " " << event.polarity << std::endl;
    }
  }

  fileOutput.close();
}