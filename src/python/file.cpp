#include "file.hpp"

void FileInput::stream_generator_to_buffer() {
  // We add a local buffer to avoid overusing the atomic
  // lock in the actual buffer
  std::vector<AER::Event> local_buffer = {};
  for (const auto event : generator) {
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
  if (local_buffer.size() > 0) {
    buffer.set_vector(local_buffer);
  }
  is_streaming.store(false);
}

FileInput::FileInput(const std::string &filename, py_size_t shape,
                     const std::string &device, bool ignore_time)
    : buffer(shape, device, EVENT_BUFFER_SIZE), ignore_time(ignore_time),
      shape(shape), filename(filename), file(open_event_file(filename)){};

std::unique_ptr<BufferPointer> FileInput::read() {
  auto tmp = buffer.read();
  is_nonempty.store(false);
  return std::unique_ptr<BufferPointer>(std::move(tmp));
}

Generator<AER::Event>::Iter FileInput::begin() { return generator.begin(); }
std::default_sentinel_t FileInput::end() { return generator.end(); }

bool FileInput::get_is_streaming() {
  return is_streaming.load() || is_nonempty.load();
}

nb::ndarray<nb::numpy, uint8_t, nb::shape<1, -1>> FileInput::load() {
  struct Container {
    std::vector<AER::Event> events;
  };
  auto [arr, n_read] = file->read_events(-1);
  Container *c = new Container();
  c->events = std::move(arr);
  nb::capsule deleter(c, [](void *p) noexcept { delete (Container *)p; });
  const size_t shape[1] = {n_read * sizeof(AER::Event)};
  return nb::ndarray<nb::numpy, uint8_t, nb::shape<1, -1>>(
      c->events.data(), 1, shape, deleter);
}

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
  generator = file->stream();
  file_thread = std::unique_ptr<std::thread>(
      new std::thread(&FileInput::stream_generator_to_buffer, this));
  return this;
}

bool FileInput::stop_stream(nb::object &a, nb::object &b, nb::object &c) {
  is_streaming.store(false);
  if (file_thread->joinable()) {
    file_thread->join();
  }
  return false;
}