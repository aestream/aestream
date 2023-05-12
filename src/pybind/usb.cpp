#include "../aer.hpp"
#include "../generator.hpp"
#include "../input/inivation.hpp"

#include "types.hpp"
#include "tensor_buffer.hpp"

class USBInput {

private:
  Generator<AER::Event> generator;
  std::thread socket_thread;
  static const uint32_t EVENT_BUFFER_SIZE = 64;
  TensorBuffer buffer;
  std::atomic<bool> is_streaming = {true};
  std::atomic<bool> done_streaming = {false};

  void stream_synchronous() {
    while (is_streaming.load()) {
      // We add a local buffer to avoid overusing the atomic lock in the actual
      // buffer
      std::vector<AER::Event> local_buffer = {};
      for (auto event : generator) {
        local_buffer.push_back(event);

        if (local_buffer.size() >= EVENT_BUFFER_SIZE) {
          buffer.set_vector(local_buffer);
          local_buffer.clear();
        }
      }
    }
    done_streaming.store(true);
  };

public:
  USBInput(py_size_t shape, const std::string& device, uint16_t deviceId,
           uint16_t deviceAddress)
      : buffer(shape, device, EVENT_BUFFER_SIZE) {
    if (deviceId > 0) {
      try {
        auto address = InivationDeviceAddress{"dvx", deviceId, deviceAddress};
        generator = inivation_event_generator(address, is_streaming);
      } catch (std::exception &e) {
        auto address = InivationDeviceAddress{"davis", deviceId, deviceAddress};
        generator = inivation_event_generator(address, is_streaming);
      }
    } else {
      generator = inivation_event_generator({}, is_streaming);
    }
  }

  std::unique_ptr<BufferPointer> read() { 
    return buffer.read(); 
  }
  void read_genn(uint32_t *bitmask, size_t size){ buffer.read_genn(bitmask, size); }

  USBInput *start_stream() {
    std::thread socket_thread(&USBInput::stream_synchronous, this);
    socket_thread.detach();
    return this;
  }

  void stop_stream() {
    is_streaming.store(false);
    while (!done_streaming.load()) {
      // Wait until the thread is done streaming to avoid freeing memory too early
    }
  }
};