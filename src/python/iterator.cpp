
#include "../aer.hpp"
#include "../generator.hpp"
#include "../input/file.hpp"
#include "file.hpp"
#include "tensor_buffer.hpp"
#include "types.hpp"

struct Iterator {
private:
  FileInput &file;
  const size_t n_events_per_part;
  nb::object ref;
  bool done = false;

public:
  Iterator(FileInput &file, size_t n_events_per_part, nb::object ref)
      : file(file), n_events_per_part(n_events_per_part), ref(ref) {}

  // Generator<AER::Event>::Iter begin() { return generator.begin(); }
  // std::default_sentinel_t end() { return generator.end(); }

  nb::ndarray<nb::numpy, AER::Event> next() {
    if (done) {
      throw nb::stop_iteration();
    }

    AER::Event *array =
        (AER::Event *)malloc(n_events_per_part * sizeof(AER::Event));

    size_t index = 0;
    for (auto event : file.generator) {
      array[index] = event;
      index++;
      if (index >= n_events_per_part) {
        const size_t s[] = {n_events_per_part};
        return nb::ndarray<nb::numpy, AER::Event>(array, 1, s);
      }
    }

    if (index > 0) {
      // return buffer_to_py_array(event_array, index);
      done = true;
      const size_t s[] = {index};
      return nb::ndarray<nb::numpy, AER::Event>(array, 1, s);
    } else {
      throw nb::stop_iteration();
    }
  }
};

struct FrameIterator {

private:
  FileInput &file;
  const size_t n_events_per_part;
  nb::object ref;
  bool done = false;

public:
  FrameIterator(FileInput &file, size_t n_events_per_part, nb::object ref)
      : file(file), n_events_per_part(n_events_per_part), ref(ref) {}

  tensor_t next() {
    if (done) {
      throw nb::stop_iteration();
    }

    auto tmp = std::vector<AER::Event>();
    size_t index = 0;
    for (auto event : file.generator) {
      tmp.push_back(event);
      index++;
      if (index >= n_events_per_part) {
        file.buffer.set_vector(tmp);
        return file.buffer.read();
      }
    }

    if (index > 0) {
      done = true;
      file.buffer.set_vector(tmp);
      return file.buffer.read();
    } else {
      throw nb::stop_iteration();
    }
  }
};

struct PartIterator {
private:
  FileInput &file;
  const size_t n_events_per_part;
  py::object ref;
  bool done = false;

public:
  PartIterator(FileInput &file, size_t n_events_per_part, py::object ref)
      : file(file), n_events_per_part(n_events_per_part), ref(ref) {}

  // Generator<AER::Event>::Iter begin() { return generator.begin(); }
  // std::default_sentinel_t end() { return generator.end(); }

  nb::ndarray<nb::numpy, AER::Event> next() {
    auto [event_array, n_events_read] =
        dat_read_n_events(file.fp, n_events_per_part);

    if (n_events_read == 0) {
      throw nb::stop_iteration();
    }

    return nb::ndarray<nb::numpy, AER::Event>(event_array, 1, n_events_read);
  }
};