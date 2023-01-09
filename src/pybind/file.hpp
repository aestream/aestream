#pragma once
#include <algorithm>

#include "../aedat.hpp"
#include "../input/file.hpp"
// #include "../aedat4.hpp"
#include "../generator.hpp"

#ifdef USE_TORCH
#include <torch/extension.h>
#include <torch/torch.h>
#endif

#include "tensor_buffer.hpp"
#include "tensor_iterator.hpp"

inline py::array_t<AER::Event> buffer_to_py_array(AER::Event *event_array,
                                                  size_t n_events) {
  py::capsule free_when_done(event_array, [](void *f) {
    AER::Event *event_array = reinterpret_cast<AER::Event *>(f);
    delete[] event_array;
  });

  return py::array_t<AER::Event>({n_events}, {sizeof(AER::Event)}, event_array,
                                 free_when_done);
}

class FileInput {

private:
  static const uint32_t EVENT_BUFFER_SIZE = 512;

  const bool ignore_time;
  py_size_t shape;

  std::unique_ptr<std::thread> file_thread;
  std::vector<AER::Event> event_vector;
  TensorBuffer buffer;
  // TensorIterator iterator;
  std::atomic<bool> is_streaming = {true};
  std::atomic<bool> is_nonempty = {true};

  void stream_file_to_buffer();

  void stream_generator_to_buffer();

public:
  const unique_file_t fp;
  Generator<AER::Event> generator;
  const std::string filename;
  size_t n_events;

  FileInput(const std::string filename, py_size_t shape, device_t device,
            bool ignore_time = false);

  tensor_t read();

  Generator<AER::Event>::Iter begin();
  std::default_sentinel_t end();

  bool get_is_streaming();

  py::array_t<AER::Event> events();

  py::array_t<AER::Event> events_co();

  // Generator<py::array_t<AER::Event>> parts_co(size_t n_events_per_part);

  FileInput *start_stream();

  void stop_stream();
};