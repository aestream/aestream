#include "../cpp/aer.hpp"
#include "../cpp/generator.hpp"
#ifdef WITH_METAVISION
#include "../cpp/input/prophesee.hpp"
#endif
#ifdef WITH_CAER
#include "../cpp/input/inivation.hpp"
#endif

#include "tensor_buffer.hpp"
#include "types.hpp"

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
  // General constructor
  USBInput(py_size_t shape, const std::string device, Camera camera)
      : buffer(shape, device, EVENT_BUFFER_SIZE) {
    if (camera == Camera::Inivation) {
#ifdef WITH_CAER
      generator = inivation_event_generator({}, is_streaming);
#else
      throw std::invalid_argument("Inivation camera drivers not available.");
#endif
    } else if (camera == Camera::Prophesee) {
#ifdef WITH_METAVISION
      generator = prophesee_event_generator(is_streaming, std::nullopt);
#else
      throw std::invalid_argument("Inivation camera drivers not available.");
#endif
    }
  }
// Inivation via LIBCAER
#ifdef WITH_CAER
  USBInput(py_size_t shape, const std::string device, uint16_t deviceId,
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
#endif
// Prophesee via Metavision
#ifdef WITH_METAVISION
  USBInput(py_size_t shape, const std::string device, const std::string serial)
      : buffer(shape, device, EVENT_BUFFER_SIZE) {
        std::cout << serial << std::endl;
    generator = prophesee_event_generator(is_streaming, serial);
  }
#endif

  std::unique_ptr<BufferPointer> read() { return buffer.read(); }
  void read_genn(uint32_t *bitmask, size_t size) {
    buffer.read_genn(bitmask, size);
  }

  USBInput *start_stream() {
    std::thread socket_thread(&USBInput::stream_synchronous, this);
    socket_thread.detach();
    return this;
  }

  void stop_stream() {
    is_streaming.store(false);
    while (!done_streaming.load()) {
      // Wait until the thread is done streaming to avoid freeing memory too
      // early
    }
  }
};
