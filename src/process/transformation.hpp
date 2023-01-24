#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include "../aedat.hpp"
#include "../generator.hpp"


struct pixel {
    int x;
    int y;
};

#define MAX_PIXIX 2

struct map {
        int np;
        pixel p[MAX_PIXIX];
};

enum trans {no_trans, rot_90, rot_180, rot_270, flip_ud, flip_lr};

trans from_string_to_trans(std::string requested_trans);

void print_lut(int width, int height, map lut[]);

void count_stuff(int width, int height, map lut[]);

void get_empty_lut(int width, int height, map lut[]);

void load_lut(const std::string & fname, int width, int height, map lut[]);

Generator<AEDAT::PolarityEvent>
transformation_event_generator(Generator<AEDAT::PolarityEvent> &input_generator,
                       const std::string &undistortion_filename, trans transformation, 
                       std::uint16_t width, std::uint16_t height, uint8_t t_sample, uint8_t s_sample);
// Generator<AEDAT::PolarityEvent> transformation_filter(Generator<AEDAT::PolarityEvent> input, std::string filename);