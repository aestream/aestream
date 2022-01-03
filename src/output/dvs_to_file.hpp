#pragma once

#include "../aedat.hpp"
#include "../generator.hpp"
#include <fstream>
#include <string>

template <typename T>
void DVSToFile(Generator<T> &input_generator, std::string filename);