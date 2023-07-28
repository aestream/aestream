#include "../aer.hpp"
#include "../generator.hpp"

struct DvsEvent {
  bool polarity;
  uint8_t y;
  uint8_t x;
  uint32_t timestamp;
};


constexpr char ZMQ_SUBSCRIBE_HEADER = 1;

Generator<AER::Event> open_zmq(const std::string socket, std::atomic<bool>& runFlag);