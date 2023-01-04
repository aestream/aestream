#include "../input/file.hpp"
#include "../aedat.hpp"
// #include "../aedat4.hpp"
#include "../generator.hpp"

#ifdef USE_TORCH
#include <torch/extension.h>
#include <torch/torch.h>
#endif

#include "tensor_buffer.hpp"

class FileInput {

private:
  static const uint32_t EVENT_BUFFER_SIZE = 512;

  const bool ignore_time;

  std::unique_ptr<std::thread> file_thread;
  Generator<AEDAT::PolarityEvent> generator;
  std::vector<AEDAT::PolarityEvent> event_vector;
  TensorBuffer buffer;
  std::atomic<bool> is_streaming = {true};
  std::atomic<bool> is_nonempty = {true};
  const bool use_coroutines;

  void stream_file_to_buffer() {
    while (is_streaming.load()) {
      const auto time_start = std::chrono::high_resolution_clock::now();
      const int64_t time_start_us = event_vector[0].timestamp;

      std::vector<AEDAT::PolarityEvent> local_buffer = {};
      for (auto event : event_vector) {
        if (!is_streaming) {
          break;
        }
        local_buffer.push_back(event);

        // Sleep to align with real-time, unless ignore_time is set
        if (!ignore_time) {
          const int64_t time_diff =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::high_resolution_clock::now() - time_start)
                  .count();
          const int64_t file_diff = event.timestamp - time_start_us;
          const int64_t time_offset = file_diff - time_diff;
          if (time_offset > 1000) {
            std::this_thread::sleep_for(std::chrono::microseconds(time_offset));
          }
        }

        if (local_buffer.size() >= EVENT_BUFFER_SIZE) {
          buffer.set_vector(local_buffer);
          is_nonempty.store(true);
          local_buffer.clear();
        }
      }
      std::cout << "Done streaming events" << std::endl;
      is_streaming.store(false);
    }
  }

  void stream_generator_to_buffer() {
    // We add a local buffer to avoid overusing the atomic
    // lock in the actual buffer
    try {
      std::vector<AEDAT::PolarityEvent> local_buffer = {};
      for (auto event : generator) {
        if (!is_streaming.load()) {
          break;
        }
        local_buffer.push_back(event);

        if (local_buffer.size() >= EVENT_BUFFER_SIZE) {
          buffer.set_vector(local_buffer);
          is_nonempty.store(true);
          local_buffer.clear();
        }
      }
      is_streaming.store(false);
    } catch (std::exception e) {
      std::cout << "caught " << e.what() << std::endl;
    }
  }

public:
  FileInput(std::string filename, py_size_t shape, device_t device,
            bool ignore_time = false, bool use_coroutines = true)
      : buffer(shape, device, EVENT_BUFFER_SIZE), ignore_time(ignore_time),
        use_coroutines(use_coroutines) {
    if (use_coroutines) {
      generator = file_event_generator(filename, is_streaming, ignore_time);
    } else {
      throw std::invalid_argument("AEDAT4 temporarily unsupported");
      // try {
      //   const AEDAT4 aedat_file = AEDAT4(filename);
      //   event_vector = aedat_file.polarity_events;
      // } catch (std::exception e) {
      //   throw std::invalid_argument(
      //       "Files without coroutines must be of .aedat4 format");
      // }
    }
  }

  tensor_t read() {
    const auto &tmp = buffer.read();
    is_nonempty.store(false);
    return tmp;
  }

  bool get_is_streaming() { return is_streaming.load() || is_nonempty.load(); }

  FileInput *start_stream() {
    if (use_coroutines) {
      file_thread = std::unique_ptr<std::thread>(
          new std::thread(&FileInput::stream_generator_to_buffer, this));
    } else {
      file_thread = std::unique_ptr<std::thread>(
          new std::thread(&FileInput::stream_file_to_buffer, this));
    }
    return this;
  }

  void stop_stream() {
    is_streaming.store(false);
    if (file_thread->joinable()) {
      file_thread->join();
    }
  }
};