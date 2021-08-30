#include <string>

#include "../aedat.hpp"
#include "../generator.hpp"

Generator<AEDAT::PolarityEvent>
file_event_generator(const std::string filename, bool ignore_time = false);