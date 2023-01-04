#include "types.hpp"

#include <pybind11/stl.h>

#include "file.cpp"
#include "udp.cpp"

#ifdef USE_INIVATION
#include "usb.cpp"
#endif

PYBIND11_MODULE(aestream_ext, m) {

  py::class_<FileInput>(m, "FileInput")
      .def(py::init<std::string, py_size_t, device_t, bool, bool>(),
           py::arg("filename"), py::arg("shape"), py::arg("device") = "cpu",
           py::arg("ignore_time") = false, py::arg("use_coroutines") = true)
      .def("__enter__", &FileInput::start_stream)
      .def("__exit__",
           [&](FileInput &i, py::object &t, py::object &v, py::object &trace) {
             i.stop_stream();
             return false;
           })
      .def("is_streaming", &FileInput::get_is_streaming)
      .def("start_stream", &FileInput::start_stream)
      .def("stop_stream", &FileInput::stop_stream)
      .def("read", &FileInput::read);

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