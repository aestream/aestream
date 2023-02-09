#include "file.hpp"

void FileInput::stream_file_to_buffer() {
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

void FileInput::stream_generator_to_buffer() {
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

FileInput::FileInput(const std::string &filename, py_size_t shape,
                    const std::string& device, bool ignore_time)
    : buffer(shape, device, EVENT_BUFFER_SIZE), ignore_time(ignore_time),
      shape(shape), filename(filename), fp(open_file(filename)) {
  if (ends_with(filename, "dat")) {
    n_events = dat_read_header(fp);
  } else {
    n_events = 0;
  }
  // if (ends_with(filename, "dat")) {
  //   generator = dat_stream_events(fp, n_events);
  // } else {
  //   generator = aedat_to_stream(filename);
  // }
  auto runFlag = std::atomic<bool>(true);
  generator = file_event_generator(filename, runFlag);
};

tensor_t FileInput::read() {
  const auto &tmp = buffer.read();
  is_nonempty.store(false);
  return tmp;
}

Generator<AER::Event>::Iter FileInput::begin() { return generator.begin(); }
std::default_sentinel_t FileInput::end() { return generator.end(); }

bool FileInput::get_is_streaming() {
  return is_streaming.load() || is_nonempty.load();
}

// nb::tensor<nb::numpy, AER::Event> FileInput::events() {
//   // const unique_file_t &fp = open_file(filename);
//   // auto n_events = dat_read_header(fp);
//   auto [event_array, n_events_read] = dat_read_n_events(fp, n_events);

//   return nb::tensor<nb::numpy, AER::Event>(event_array, 1, n_events_read);
// }

// py::array_t<AER::Event> FileInput::events_co() {
//   AER::Event *event_array = (AER::Event *)malloc(n_events *
//   sizeof(AER::Event)); size_t index = 0; for (auto event : generator) {
//     event_array[index] = event;
//     index++;
//   }
//   return buffer_to_py_array(event_array, n_events);
// }

// Generator<py::array_t<AER::Event>>
// FileInput::parts_co(size_t n_events_per_part) {
//   // generator = dat_stream_events(fp);

//   AER::Event *event_array =
//       (AER::Event *)malloc(n_events_per_part * sizeof(AER::Event));
//   // auto event_array = py::array_t<AER::Event>(n_events_per_part);
//   // AER::Event *event_ptr =
//   //     static_cast<AER::Event *>(event_array.request().ptr);
//   size_t index = 0;
//   size_t part_index = 0;
//   for (auto event : generator) {
//     event_array[index] = event;
//     index++;
//     if (index % n_events_per_part == 0) {
//       std::cout << "Produced " << index << std::endl;
//       co_yield buffer_to_py_array(event_array, n_events_per_part);
//       // event_array = py::array_t<AER::Event>(
//       //     std::min(n_events_per_part, n_events - part_index));
//       event_array =
//           (AER::Event *)malloc(n_events_per_part * sizeof(AER::Event));
//       // event_array.reset(
//       //     (AER::Event *)malloc(n_events_per_part *
//       //     sizeof(AER::Event)));
//       index = 0;
//     }
//   }
//   // if (index % n_events_per_part > 0) {
//   //   co_yield buffer_to_py_array(events, index % n_events_per_part);
//   // }
// }

FileInput *FileInput::start_stream() {
  // generator = file_event_generator(filename, is_streaming);
  file_thread = std::unique_ptr<std::thread>(
      new std::thread(&FileInput::stream_generator_to_buffer, this));
  return this;
}

void FileInput::stop_stream() {
  is_streaming.store(false);
  if (file_thread->joinable()) {
    file_thread->join();
  }
}