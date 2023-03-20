
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "../aer.hpp"

#include "file.hpp"
// #include "iterator.cpp"
#include "types.hpp"
#include "udp.cpp"

#ifdef USE_INIVATION
#include "usb.cpp"
#endif

namespace nb = nanobind;

NB_MODULE(aestream_ext, m) {

  nb::class_<AER::Event>(m, "Event")
      .def(nb::init<uint64_t, uint16_t, uint16_t, bool>())
      .def_rw("timestamp", &AER::Event::timestamp)
      .def_rw("x", &AER::Event::x)
      .def_rw("y", &AER::Event::y)
      .def_rw("polarity", &AER::Event::polarity);

  nb::class_<BufferPointer>(m, "BufferPointer")
      .def("to_numpy", &BufferPointer::to_numpy)
      .def("to_torch", &BufferPointer::to_torch);

  //   nb::class_<Iterator>(m, "EventIterator")
  //       //  .def( // Thanks to https://stackoverflow.com/a/57217995/999865
  //       // "__iter__",
  //       // [](Iterator &f) { return nb::make_iterator(f.begin(), f.end()); },
  //       // nb::keep_alive<0, 1>() /* Keep object alive while iterator exists
  //       //                         */
  //       // ,
  //       // nb::return_value_policy::move)
  //       .def("__iter__", [](Iterator &it) -> Iterator & { return it; })
  //       .def("__next__", &Iterator::next, nb::rv_policy::reference);

  //   nb::class_<FrameIterator>(m , "FrameIterator")
  //       .def("__iter__", [](FrameIterator &it) -> FrameIterator & { return
  //       it; }) .def("__next__", &FrameIterator::next);

  //   nb::class_<PartIterator>(m, "PartIterator")
  //       .def("__iter__", [](PartIterator &it) -> PartIterator & { return it;
  //       }) .def("__next__", &PartIterator::next);

  nb::class_<FileInput>(m, "FileInput")
      .def(nb::init<std::string, py_size_t, std::string, bool>(),
           nb::arg("filename"), nb::arg("shape"), nb::arg("device") = "cpu",
           nb::arg("ignore_time") = false)
      .def("__enter__", &FileInput::start_stream)
      .def("__exit__", &FileInput::stop_stream, nb::arg("a").none(),
           nb::arg("b").none(), nb::arg("c").none())
      .def("load_all", &FileInput::load)
      //  .def("frames",
      //       [](nb::object fobj, size_t n_events_per_part) {
      //         return FrameIterator(fobj.cast<FileInput &>(),
      //         n_events_per_part, fobj);
      //       })
      //  .def("events_co", &FileInput::events_co)
      .def("is_streaming", &FileInput::get_is_streaming)
      .def("start_stream", &FileInput::start_stream)
      .def("stop_stream", &FileInput::stop_stream)
      .def("read_buffer", &FileInput::read);
  //  .def("parts",
  //       [](nb::object fobj, size_t n_events_per_part) {
  //         //    std::cout << i.filename << std::endl;
  //         //    if (endsWith(i.filename, "aedat4")) {
  //         // std::cout << "AEDAT" << std::endl;
  //         return Iterator(fobj.cast<FileInput &>(), n_events_per_part,
  //         fobj);
  //         //    } else {
  //         // return PartIterator(i, n_events_per_part, fobj);
  //         //    }
  //       })
  //  .def( // Thanks to https://stackoverflow.com/a/57217995/999865
  //      "__iter__",
  //      [](FileInput &f) { return
  //      nb::make_iterator(nb::type<Generator<AER::Event>::Iter>,
  //      "iterator", f.begin(), f.end()); }, nb::keep_alive<0, 1>() /* Keep
  //      object alive while iterator exists */)
  //  .def("__next__", &FileInput::begin)
  ;

  nb::class_<UDPInput>(m, "UDPInput")
      .def(nb::init<py_size_t, std::string, int>(), nb::arg("shape"),
           nb::arg("device") = "cpu", nb::arg("port") = 3333)
      .def("__enter__", &UDPInput::start_stream)
      .def("__exit__", &UDPInput::stop_stream, nb::arg("a").none(),
           nb::arg("b").none(), nb::arg("c").none())
      .def("start_stream", &UDPInput::start_stream)
      .def("stop_stream", &UDPInput::stop_stream, nb::arg("a").none(),
           nb::arg("b").none(), nb::arg("c").none())
      .def("read_buffer", &UDPInput::read);

#ifdef USE_INIVATION
  nb::class_<USBInput>(m, "USBInput")
      .def(nb::init<py_size_t, device_t, int, int>(), nb::arg("shape"),
           nb::arg("device") = "cpu", nb::arg("device_id") = 0,
           nb::arg("device_address") = 0)
      .def("__enter__", &USBInput::start_stream)
      .def("__exit__",
           [&](USBInput &i, nb::object &t, nb::object &v, nb::object &trace) {
             i.stop_stream();
             return false;
           })
      .def("start_stream", &USBInput::start_stream)
      .def("stop_stream", &USBInput::stop_stream)
      .def("read", &USBInput::read);
#endif
}