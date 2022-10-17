#include <string>
#include <torch/extension.h>
#include <torch/torch.h>

#include "udp.cpp"
#include "usb.cpp"


PYBIND11_MODULE(aestream_ext, m) {

  py::class_<USBInput>(m, "USBInput")
      .def(py::init<torch::IntArrayRef, torch::Device, int, int>(),
           py::arg("shape"), py::arg("device") = "cpu",
           py::arg("device_id") = 0, py::arg("device_address") = 0)
      .def(py::init<torch::IntArrayRef, std::string, int, int>(),
           py::arg("shape"), py::arg("device") = "cpu",
           py::arg("device_id") = 0, py::arg("device_address") = 0)
      .def("__enter__", &USBInput::start_stream)
      .def("__exit__",
           [&](USBInput &i, py::object t, py::object v, py::object trace) {
             i.stop_stream();
             return false;
           })
      .def("start_stream", &USBInput::start_stream)
      .def("stop_stream", &USBInput::stop_stream)
      .def("read", &USBInput::read);

  py::class_<UDPInput>(m, "UDPInput")
      .def(py::init<torch::IntArrayRef, torch::Device, int>(), py::arg("shape"),
           py::arg("device") = "cpu", py::arg("port") = 3333)
      .def("__enter__", &UDPInput::start_server)
      .def("__exit__",
           [&](UDPInput &i, py::object t, py::object v, py::object trace) {
             i.stop_server();
             return false;
           })
      .def("start_stream", &UDPInput::start_server)
      .def("stop_stream", &UDPInput::stop_server)
      .def("read", &UDPInput::read);
}