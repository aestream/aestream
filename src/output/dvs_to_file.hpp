#pragma once

#include <fstream>
#include <string>

#include "../aer.hpp"
#include "../file/aedat.hpp"
#include "../file/aedat4.hpp"
#include "../generator.hpp"

void dvs_to_file_aedat(Generator<AER::Event> &input_generator,
                       const std::string &filename,
                       size_t bufferSize = 1 << 12);

void dvs_to_file_csv(Generator<AER::Event> &input_generator,
                     const std::string &filename);