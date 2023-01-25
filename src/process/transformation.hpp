#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

#include "../aer.hpp"
#include "../generator.hpp"


#define MAX_PIXIX 2

struct lutmap {
        uint8_t np;
        int x[MAX_PIXIX];
        int y[MAX_PIXIX];
};

enum trans {no_trans, rot_90, rot_180, rot_270, flip_ud, flip_lr};


trans from_string_to_trans(std::string requested_trans);

void load_lut(const std::string & fname, int width, int height, lutmap lut[]);

Generator<AER::Event>
transformation_event_generator(Generator<AER::Event> &input_generator,
                       const std::string &undistortion_filename, trans transformation, 
                       std::uint16_t width, std::uint16_t height, uint8_t t_sample, uint8_t s_sample);
