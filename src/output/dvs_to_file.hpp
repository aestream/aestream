#pragma once

#include <string>

#include "../aedat.hpp"
#include "../generator.hpp"

void dvs_to_file_aedat(Generator<AEDAT::PolarityEvent> &input_generator,
                       const std::string &filename,
                       size_t bufferSize = 1 << 12);

void dvs_to_file_txt(Generator<AEDAT::PolarityEvent> &input_generator,
                     const std::string &filename);