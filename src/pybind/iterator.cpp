
#include "../aer.hpp"
#include "../generator.hpp"
#include "../input/file.hpp"
#include "file.hpp"
#include "types.hpp"

struct Iterator {
private:
  FileInput &file;
  const size_t n_events_per_part;
  py::object ref;
  bool done = false;

public:
  Iterator(FileInput &file, size_t n_events_per_part, py::object ref)
      : file(file), n_events_per_part(n_events_per_part), ref(ref) {}

  // Generator<AER::Event>::Iter begin() { return generator.begin(); }
  // std::default_sentinel_t end() { return generator.end(); }

  py::array_t<AER::Event> next() {
    if (done) {
      throw py::stop_iteration();
    }

    auto array = py::array_t<AER::Event>(n_events_per_part);
    auto event_array = (AER::Event *)array.request().ptr;

    size_t index = 0;
    for (auto event : file.generator) {
      event_array[index] = event;
      index++;
      if (index >= n_events_per_part) {
        return array;
      }
    }

    if (index > 0) {
      // return buffer_to_py_array(event_array, index);
      done = true;
      return array;
    } else {
      throw py::stop_iteration();
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

  py::array_t<AER::Event> next() {
    auto [event_array, n_events_read] =
        dat_read_n_events(file.fp, n_events_per_part);

    if (n_events_read == 0) {
      throw py::stop_iteration();
    }

    return buffer_to_py_array(event_array, n_events_read);
  }
};