#pragma once

#include <fstream>
#include <string>

#include "aestream/aer.hpp"
#include "aestream/file/aedat.hpp"
#include "aestream/file/aedat4.hpp"
#include "aestream/generator.hpp"

void dvs_to_file_aedat(Generator<AER::Event> &input_generator,
                       const std::string &filename,
                       size_t bufferSize = 1 << 12);

void dvs_to_file_txt(Generator<AER::Event> &input_generator,
                     const std::string &filename);