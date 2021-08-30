#pragma once

#include <string>

#include "../aedat.hpp"
#include "../generator.hpp"

void dvs_to_file(Generator<AEDAT::PolarityEvent> &input_generator,
                 const std::string &filename);