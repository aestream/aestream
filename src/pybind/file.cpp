#include "../input/file.hpp"
#include "../aedat.hpp"
// #include "../aedat4.hpp"
#include "../generator.hpp"

#ifdef USE_TORCH
#include <torch/extension.h>
#include <torch/torch.h>
#endif

#include "tensor_buffer.hpp"
#include "tensor_iterator.hpp"

inline py::array_t<AER::Event> buffer_to_py_array(AER::Event *events,
                                                  size_t n_events) {
  py::capsule free_when_done(events, [](void *f) {
    AER::Event *events = reinterpret_cast<AER::Event *>(f);
    delete[] events;
  });

  // return py::array(n_events, events);
  return py::array_t<AER::Event>({n_events}, {sizeof(AER::Event)}, events,
                                 free_when_done);
}
class FileInput {

private:
  static const uint32_t EVENT_BUFFER_SIZE = 512;

  const std::string filename;
  const bool ignore_time;
  py_size_t shape;

  std::unique_ptr<std::thread> file_thread;
  std::vector<AER::Event> event_vector;
  TensorBuffer buffer;
  // TensorIterator iterator;
  std::atomic<bool> is_streaming = {true};
  std::atomic<bool> is_nonempty = {true};

  void stream_file_to_buffer() {
    while (is_streaming.load()) {
      const auto time_start = std::chrono::high_resolution_clock::now();
      const int64_t time_start_us = event_vector[0].timestamp;

      std::vector<AER::Event> local_buffer = {};
      for (auto event : event_vector) {
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

          if (!is_streaming) {
            break;
          }
        }
      }
      // Stream remaining events
      if (local_buffer.size() > 0) {
        buffer.set_vector(local_buffer);
        is_nonempty.store(true);
      }
      is_streaming.store(false);
    }
  }

  void stream_generator_to_buffer() {
    // We add a local buffer to avoid overusing the atomic
    // lock in the actual buffer
    std::vector<AER::Event> local_buffer = {};
    for (const auto &event : generator) {
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
  }

public:
  Generator<AER::Event> generator;

  FileInput(const std::string filename, py_size_t shape, device_t device,
            bool ignore_time = false)
      : buffer(shape, device, EVENT_BUFFER_SIZE), ignore_time(ignore_time),
        shape(shape), filename(filename){};

  tensor_t read() {
    const auto &tmp = buffer.read();
    is_nonempty.store(false);
    return tmp;
  }

  Generator<AER::Event>::Iter begin() { return generator.begin(); }
  std::default_sentinel_t end() { return generator.end(); }

  bool get_is_streaming() { return is_streaming.load() || is_nonempty.load(); }

  // std::vector<AEDAT::PolarityEvent> events() {
  py::array_t<AER::Event> events() {
    const unique_file_t &fp = open_file(filename);

    auto n_events = dat_read_header(fp);
    auto events = dat_read_all_events(fp, n_events);

    return buffer_to_py_array(events, n_events);
  }

  py::array_t<AER::Event> events_co() {
    const unique_file_t &fp = open_file(filename);
    auto n_events = dat_read_header(fp);
    generator = dat_read_stream_all_events(fp);

    AER::Event *events = (AER::Event *)malloc(n_events * sizeof(AER::Event));
    size_t index = 0;
    for (auto event : generator) {
      events[index] = event;
      index++;
    }
    return buffer_to_py_array(events, n_events);
  }

  FileInput *start_stream() {
    generator = file_event_generator(filename, is_streaming, ignore_time);
    file_thread = std::unique_ptr<std::thread>(
        new std::thread(&FileInput::stream_generator_to_buffer, this));
    return this;
  }

  void stop_stream() {
    is_streaming.store(false);
    if (file_thread->joinable()) {
      file_thread->join();
    }
  }
};