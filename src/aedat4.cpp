#include <string>

#include "aedat.hpp"
#include "aedat4.hpp"

Generator<AEDAT::PolarityEvent> file_event_generator(std::string filename) {
    AEDAT4 aedat_file = AEDAT4(filename);
    auto polarity_events = aedat_file.polarity_events;
    int index = 0;
    while (index < polarity_events.size()) {
        AEDAT::PolarityEvent event = polarity_events[index];
        co_yield event;
        index++;
    }
}