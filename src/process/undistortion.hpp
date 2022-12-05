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

struct map {
        int np;
        pixel p[2];
};

void print_lut(int width, int height, map lut[]);

void count_stuff(int width, int height, map lut[]);

void get_empty_lut(int width, int height, map lut[]);

void load_lut(const std::string & fname, int width, int height, map lut[]);

Generator<AEDAT::PolarityEvent>
undistortion_event_generator(Generator<AEDAT::PolarityEvent> &input_generator,
                       const std::string &filename, const std::uint16_t width, const std::uint16_t height);
// Generator<AEDAT::PolarityEvent> undistortion_filter(Generator<AEDAT::PolarityEvent> input, std::string filename);