#pragma once
#include <algorithm>

#include "../aer.hpp"
#include "../file/aedat4.hpp"
#include "../generator.hpp"
#include "../input/file.hpp"

#include "tensor_buffer.hpp"
#include "tensor_iterator.hpp"

class FileInput {

private:
  static const uint32_t EVENT_BUFFER_SIZE = 512;

  const bool ignore_time;

  std::unique_ptr<std::thread> file_thread;
  std::vector<AER::Event> event_vector;
  // TensorIterator iterator;
  std::atomic<bool> is_streaming = {true};
  std::atomic<bool> is_nonempty = {true};

  void stream_file_to_buffer();

  void stream_generator_to_buffer();

public:
  TensorBuffer buffer;
  py_size_t shape;
  const std::unique_ptr<FileBase> file;
  Generator<AER::Event> generator;
  const std::string filename;

  FileInput(const std::string &filename, py_size_t shape,
            const std::string &device, bool ignore_time = false);

  BufferPointer read();

  Generator<AER::Event>::Iter begin();
  std::default_sentinel_t end();

  bool get_is_streaming();

  nb::ndarray<nb::numpy, uint8_t, nb::shape<1, nb::any>> load();

  // py::array_t<AER::Event> events_co();

  // Generator<py::array_t<AER::Event>> parts_co(size_t n_events_per_part);

  FileInput *start_stream();

  bool stop_stream(nb::object &a, nb::object &b, nb::object &c);
};