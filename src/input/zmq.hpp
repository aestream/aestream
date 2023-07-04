#include "../aer.hpp"
#include "../generator.hpp"

struct DvsEvent {
  unsigned int polarity : 1;
  unsigned int y : 8;
  unsigned int x : 8;
  unsigned int timestamp : 32;
} __attribute__((packed));

constexpr char ZMQ_SUBSCRIBE_HEADER = 1;

Generator<AER::Event> open_zmq(const std::string socket, std::atomic<bool>& runFlag);