#include <string>
#include <atomic>

#include "../aedat.hpp"
#include "../generator.hpp"

/**
 * Reads AEDAT events from a file and replays them either in real-time
 * (ignore_time = false) or as fast as possible (ignore_time = true).
 *
 * @param filename  The path to the file
 * @param ignore_time  Whether to ignore the timestamps and replay the events as
 * fast as possible (true) or enforce that the events are replayed in real-time
 * (false, default).
 * @return A Generator of PolarityEvents
 */
Generator<AEDAT::PolarityEvent> file_event_generator(const std::string filename,
                                                     const std::atomic<bool> &runFlag,
                                                     bool ignore_time = false);