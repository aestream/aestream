#include "../aer.hpp"
#include "../generator.hpp"

Generator<AER::Event> open_zmq(const std::string socket, std::atomic<bool>& runFlag);