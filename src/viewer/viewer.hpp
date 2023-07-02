#include "../aer.hpp"
#include "../generator.hpp"

int view_stream(Generator<AER::Event> &generator, size_t width, size_t height,
                size_t frame_duration, bool quiet);