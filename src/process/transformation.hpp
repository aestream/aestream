#pragma once

#include <string.h>

#include "../aer.hpp"
#include "../generator.hpp"

enum trans {no_trans, rot_90, rot_180, rot_270, flip_ud, flip_lr};

trans from_string_to_trans(const std::string &requested_trans);

Generator<AER::Event>
transformation_event_generator(Generator<AER::Event> &input_generator,
                       const std::string &undistortion_filename, trans transformation, 
                       uint16_t width, uint16_t height, uint8_t t_sample, uint8_t s_sample);
