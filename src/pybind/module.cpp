#include <string>
#include <torch/extension.h>
#include <torch/torch.h>

#include "udp.cpp"
#include "usb.cpp"
#include "file.cpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(aestream, m) {
  py::class_<DVSInput>(m, "DVSInput")
      .def(py::init<int, int, torch::IntArrayRef, torch::Device>(),
           py::arg("device_id"), py::arg("device_address"), py::arg("shape"),
           py::arg("device") = torch::DeviceType::CPU)
      .def(py::init<int, int, torch::IntArrayRef, std::string>(),
           py::arg("device_id"), py::arg("device_address"), py::arg("shape"),
           py::arg("device") = "cpu")
      .def("__enter__", &DVSInput::start_stream)
      .def("__exit__",
           [&](DVSInput &i, py::object t, py::object v, py::object trace) {
             i.stop_stream();
             return false;
           })
      .def("start_stream", &DVSInput::start_stream)
      .def("stop_stream", &DVSInput::stop_stream)
      .def("read", &DVSInput::read);

  py::class_<UDPInput>(m, "UDPInput")
      .def(py::init<torch::IntArrayRef, torch::Device, int>(), py::arg("shape"),
           py::arg("device") = torch::DeviceType::CPU, py::arg("port") = 3333)
      .def(py::init<torch::IntArrayRef, std::string, int>(), py::arg("shape"),
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

  py::class_<FileInput>(m, "FileInput")
      .def(py::init<std::string, torch::IntArrayRef, torch::Device>(),
           py::arg("filename"), py::arg("shape"),
           py::arg("device") = torch::DeviceType::CPU)
      .def(py::init<std::string, torch::IntArrayRef, std::string>(),
           py::arg("filename"), py::arg("shape"),
           py::arg("device") = "cpu")
      .def("__enter__", &FileInput::start_stream)
      .def("__exit__",
           [&](FileInput &i, py::object t, py::object v, py::object trace) {
             i.stop_stream();
             return false;
           })
      .def("start_stream", &FileInput::start_stream)
      .def("stop_stream", &FileInput::stop_stream)
      .def("read", &FileInput::read);
}
