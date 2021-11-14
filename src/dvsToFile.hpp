#ifndef DVSTOFILE_HPP
#define DVSTOFILE_HPP

#include <string>
#include <fstream>
#include "aedat.hpp"
#include "generator.hpp"

template <typename T>
void DVSToFile(Generator<T>& input_generator, std::string filename);

#endif