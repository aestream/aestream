
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "../aer.hpp"

#include "file.hpp"
#include "iterator.cpp"
#include "types.hpp"
#include "udp.cpp"

#ifdef USE_INIVATION
#include "usb.cpp"
#endif

PYBIND11_MODULE(aestream_ext, m) {

  PYBIND11_NUMPY_DTYPE(AER::Event, timestamp, x, y, polarity);

  py::class_<AER::Event>(m, "Event")
      .def_property_readonly("timestamp",
                             [](const AER::Event &e) { return e.timestamp; })
      .def_property_readonly("x", [](const AER::Event &e) { return e.x; })
      .def_property_readonly("y", [](const AER::Event &e) { return e.y; })
      .def_property_readonly("polarity",
                             [](const AER::Event &e) { return e.polarity; });

  py::class_<Iterator>(m, "EventIterator")
      //  .def( // Thanks to https://stackoverflow.com/a/57217995/999865
      // "__iter__",
      // [](Iterator &f) { return py::make_iterator(f.begin(), f.end()); },
      // py::keep_alive<0, 1>() /* Keep object alive while iterator exists
      //                         */
      // ,
      // py::return_value_policy::move)
      .def("__iter__", [](Iterator &it) -> Iterator & { return it; })
      .def("__next__", &Iterator::next, py::return_value_policy::reference);

  py::class_<FrameIterator>(m, "FrameIterator")
      .def("__iter__", [](FrameIterator &it) -> FrameIterator & { return it; })
      .def("__next__", &FrameIterator::next);

  py::class_<PartIterator>(m, "PartIterator")
      .def("__iter__", [](PartIterator &it) -> PartIterator & { return it; })
      .def("__next__", &PartIterator::next);

  py::class_<FileInput>(m, "FileInput")
      .def(py::init<std::string, py_size_t, device_t, bool>(),
           py::arg("filename"), py::arg("shape"), py::arg("device") = "cpu",
           py::arg("ignore_time") = false)
      .def("__enter__", &FileInput::start_stream)
      .def("__exit__",
           [&](FileInput &i, py::object &t, py::object &v, py::object &trace) {
             i.stop_stream();
             return false;
           })
      .def("events", &FileInput::events)
      .def("frames",
           [](py::object fobj, size_t n_events_per_part) {
             return FrameIterator(fobj.cast<FileInput &>(), n_events_per_part, fobj);
           })
      //  .def("events_co", &FileInput::events_co)
      .def("is_streaming", &FileInput::get_is_streaming)
      .def("start_stream", &FileInput::start_stream)
      .def("stop_stream", &FileInput::stop_stream)
      .def("read", &FileInput::read)
      .def("parts",
           [](py::object fobj, size_t n_events_per_part) {
             //    std::cout << i.filename << std::endl;
             //    if (endsWith(i.filename, "aedat4")) {
             // std::cout << "AEDAT" << std::endl;
             return Iterator(fobj.cast<FileInput &>(), n_events_per_part, fobj);
             //    } else {
             // return PartIterator(i, n_events_per_part, fobj);
             //    }
           })
      .def( // Thanks to https://stackoverflow.com/a/57217995/999865
          "__iter__",
          [](FileInput &f) { return py::make_iterator(f.begin(), f.end()); },
          py::keep_alive<0, 1>() /* Keep object alive while iterator exists */)
      .def("__next__", &FileInput::begin);

  py::class_<UDPInput>(m, "UDPInput")
      .def(py::init<py_size_t, device_t, int>(), py::arg("shape"),
           py::arg("device") = "cpu", py::arg("port") = 3333)
      .def("__enter__", &UDPInput::start_server)
      .def("__exit__",
           [&](UDPInput &i, py::object &t, py::object &v, py::object &trace) {
             i.stop_server();
             return false;
           })
      .def("start_stream", &UDPInput::start_server)
      .def("stop_stream", &UDPInput::stop_server)
      .def("read", &UDPInput::read);

#ifdef USE_INIVATION
  py::class_<USBInput>(m, "USBInput")
      .def(py::init<py_size_t, device_t, int, int>(), py::arg("shape"),
           py::arg("device") = "cpu", py::arg("device_id") = 0,
           py::arg("device_address") = 0)
      .def("__enter__", &USBInput::start_stream)
      .def("__exit__",
           [&](USBInput &i, py::object &t, py::object &v, py::object &trace) {
             i.stop_stream();
             return false;
           })
      .def("start_stream", &USBInput::start_stream)
      .def("stop_stream", &USBInput::stop_stream)
      .def("read", &USBInput::read);
#endif
}