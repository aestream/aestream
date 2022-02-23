#include "../aedat.hpp"
#include "../generator.hpp"
#include "../input/inivation.hpp"

#include <torch/extension.h>
#include <torch/torch.h>

#include "tensor_buffer.hpp"

class DVSInput {

private:
  Generator<AEDAT::PolarityEvent> generator;
  std::thread socket_thread;
  static const uint32_t EVENT_BUFFER_SIZE = 100;
  TensorBuffer buffer;
  std::atomic<bool> is_streaming = {true};

  void stream_synchronous() {
    while (is_streaming.load()) {
      // We add a local buffer to avoid overusing the atomic lock in the actual
      // buffer
      std::vector<AEDAT::PolarityEvent> local_buffer = {};
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
  DVSInput(uint16_t deviceId, uint8_t deviceAddress, torch::IntArrayRef shape,
           std::string device)
      : buffer(shape, device) {
    generator = inivation_event_generator("davis", deviceId, deviceAddress);
  }

  at::Tensor read() { return buffer.read(); }

  DVSInput *start_stream() {
    std::thread socket_thread(&DVSInput::stream_synchronous, this);
    socket_thread.detach();
    return this;
  }

  void stop_stream() { is_streaming.store(false); }
};
