#include "input/zmq.hpp"
#include "types.hpp"
#include "tensor_buffer.hpp"

struct ZmqInput {

  ZmqInput(py_size_t shape, const std::string& device, const std::string& address)
      : buffer(shape, device, EVENT_BUFFER_SIZE), address(address) {
  }

  std::unique_ptr<BufferPointer> read() { 
    return buffer.read(); 
  }
  void read_genn(uint32_t *bitmask, size_t size){ buffer.read_genn(bitmask, size); }

  void start_stream() {
    generator = open_zmq(address, is_streaming);
  }

  void stop_stream() {
    is_streaming.store(false);
  }

private:
  const std::string address;
  Generator<AER::Event> generator;
  std::atomic<bool> is_streaming = {true};
  static const uint32_t EVENT_BUFFER_SIZE = 64;

}