#include "../aer.hpp"
#include "../generator.hpp"
#include "../input/inivation.hpp"

#include "types.hpp"

#include "tensor_buffer.hpp"

class USBInput {

private:
  Generator<AER::Event> generator;
  std::thread socket_thread;
  static const uint32_t EVENT_BUFFER_SIZE = 128;
  TensorBuffer buffer;
  std::atomic<bool> is_streaming = {true};

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

  tensor_t read() { return buffer.read(); }

  USBInput *start_stream() {
    std::thread socket_thread(&USBInput::stream_synchronous, this);
    socket_thread.detach();
    return this;
  }

  void stop_stream() {
    is_streaming.store(false);
    socket_thread.join();
  }
};