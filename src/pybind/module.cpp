#include <string>
#include <torch/extension.h>
#include <torch/torch.h>

#include "udp.cpp"
#include "usb.cpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(aestream, m) {
  py::class_<DVSInput>(m, "DVSInput")
      .def(py::init<int, int, torch::IntArrayRef, std::string>(),
           py::arg("device_id"), py::arg("device_address"), py::arg("shape"),
           py::arg("device") = "cpu")
      .def("__enter__", &DVSInput::start_stream)
      .def("__exit__",
           [&](DVSInput &i, py::object t, py::object v, py::object trace) {
             i.stop_stream();
             return true;
           })
      .def("read", &DVSInput::read);

  py::class_<UDPInput>(m, "UDPInput")
      .def(py::init<torch::IntArrayRef, std::string, int>(), py::arg("shape"),
           py::arg("device") = "cpu", py::arg("port") = 3333)
      .def("__enter__", &UDPInput::start_server)
      .def("__exit__",
           [&](UDPInput &i, py::object t, py::object v, py::object trace) {
             i.stop_server();
             return true;
           })
      .def("read", &UDPInput::read);
}