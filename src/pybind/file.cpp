#include "../aedat.hpp"
#include "../generator.hpp"
#include "../input/file.hpp"

#include <torch/extension.h>
#include <torch/torch.h>

#include "tensor_buffer.hpp"

class FileInput {

private:
  TensorBuffer buffer;
  Generator<AEDAT::PolarityEvent> generator;
  std::thread file_thread;
  static const uint32_t EVENT_BUFFER_SIZE = 100;
  std::atomic<bool> is_streaming = {true};

  void stream_synchronous() {
    // We add a local buffer to avoid overusing the atomic lock in the actual
    // buffer
    std::vector<AEDAT::PolarityEvent> local_buffer = {};
    for (auto event : generator) {
        if (!is_streaming.load()) break;
        local_buffer.push_back(event);

        if (local_buffer.size() >= EVENT_BUFFER_SIZE) {
          buffer.set_vector(local_buffer);
          local_buffer.clear();
        }
    }
  };

public:
  FileInput(const std::string filename, torch::IntArrayRef shape,
           torch::Device device)
      : buffer{shape, device},
      generator{file_event_generator(filename, is_streaming)},
      file_thread(&FileInput::stream_synchronous, this)
  {}

  at::Tensor read() { return buffer.read(); }

  FileInput *start_stream() { return this; }

  void stop_stream() {
    is_streaming.store(false);
    file_thread.join();
  }
};
