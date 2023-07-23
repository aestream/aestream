#include "input/zmq.hpp"
#include "types.hpp"
#include "tensor_buffer.hpp"

struct ZMQInput {

  ZMQInput(py_size_t shape, const std::string& device, const std::string& address)
      : buffer(shape, device, EVENT_BUFFER_SIZE), address(address) {
  }

  std::unique_ptr<BufferPointer> read() { 
    return buffer.read(); 
  }
  void read_genn(uint32_t *bitmask, size_t size){ buffer.read_genn(bitmask, size); }

  ZMQInput *start_stream() {
    generator = open_zmq(address, is_streaming);
    std::thread socket_thread(&ZMQInput::stream_synchronous, this);
    socket_thread.detach();
    return this;
  }

  void stop_stream() {
    is_streaming.store(false);
  }

private:
  const std::string address;
  Generator<AER::Event> generator;
  TensorBuffer buffer;
  std::atomic<bool> is_streaming = {true};
  static const uint32_t EVENT_BUFFER_SIZE = 64;

  void stream_synchronous() {
    while (is_streaming.load()) {
      // We add a local buffer to avoid overusing the atomic lock in the actual
      // buffer
      std::vector<AER::Event> local_buffer = {};
      for (auto event : generator) {
        if (event.x > buffer.shape[0] || event.y > buffer.shape[1]) {
          continue;
        }
        local_buffer.push_back(event);

        if (local_buffer.size() >= EVENT_BUFFER_SIZE) {
          buffer.set_vector(local_buffer);
          local_buffer.clear();
        }
      }
    }
    is_streaming.store(false);
  };
};