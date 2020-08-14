#include "convert.hpp"
#include "dvs_gesture.hpp"

#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <cstddef>
#include <sys/types.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <type_traits>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

torch::Tensor
convert_polarity_events(std::vector<AEDAT::PolarityEvent> &polarity_events,
                        const std::vector<int64_t> &tensor_size) {
  const size_t size = polarity_events.size();
  std::vector<int64_t> indices(3 * size);
  std::vector<int8_t> values;
  const auto max_duration =
      tensor_size.empty()
          ? polarity_events.back().timestamp - polarity_events[0].timestamp
          : tensor_size[0];

  for (size_t idx = 0; idx < size; idx++) {
    auto event = polarity_events[idx];
    auto event_time = event.timestamp - polarity_events[0].timestamp;
    //  Break if event is after max_duration
    if (event_time >= max_duration) {
      break;
    }

    indices[idx] = event_time;
    indices[size + idx] = event.x;
    indices[2 * size + idx] = event.y;
    values.push_back(event.polarity ? 1 : -1);
  }

  auto index_options = torch::TensorOptions().dtype(torch::kInt64);
  torch::Tensor ind = torch::from_blob(
      indices.data(), {3, static_cast<uint32_t>(size)}, index_options);

  auto value_options = torch::TensorOptions().dtype(torch::kInt8);
  torch::Tensor val = torch::from_blob(
      values.data(), {static_cast<uint32_t>(size)}, value_options);

  auto events =
      tensor_size.empty()
          ? torch::sparse_coo_tensor(ind, val)
          : torch::sparse_coo_tensor(ind, val, torch::IntArrayRef(tensor_size));

  return events.clone();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<AEDAT::PolarityEvent>(m, "PolarityEvent");

  py::class_<dvs_gesture::DataSet::DataPoint>(m, "DVSGestureDataPoint")
      .def_readonly("label", &dvs_gesture::DataSet::DataPoint::label)
      .def_readonly("events", &dvs_gesture::DataSet::DataPoint::events);

  py::class_<dvs_gesture::DataSet>(m, "DVSGestureData")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &>())
      .def("load", &dvs_gesture::DataSet::load)
      .def_readonly("datapoints", &dvs_gesture::DataSet::datapoints);

  py::class_<AEDAT4::Frame>(m, "AEDAT4Frame")
      .def_readwrite("time", &AEDAT4::Frame::time)
      .def_readwrite("width", &AEDAT4::Frame::width)
      .def_readwrite("height", &AEDAT4::Frame::height)
      .def_readwrite("pixels", &AEDAT4::Frame::pixels);

  py::class_<AEDAT>(m, "AEDAT")
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def("load", &AEDAT::load)
      .def_readwrite("polarity_events", &AEDAT::polarity_events)
      .def_readwrite("dynapse_events", &AEDAT::dynapse_events)
      .def_readwrite("imu6_events", &AEDAT::imu6_events)
      .def_readwrite("imu9_events", &AEDAT::imu9_events);

  m.def("convert_polarity_events", &convert_polarity_events,
        py::arg("polarity_events"),
        py::arg("tensor_size") = std::vector<int64_t>(),
        "Converts the AEDAT data into a sparse Torch tensor. If provided, the "
        "tensor is loaded and shaped after the tensor_size argument");

  py::class_<AEDAT4>(m, "AEDAT4")
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def("load", &AEDAT4::load)
      .def_readwrite("polarity_events", &AEDAT4::polarity_events)
      .def_readwrite("frames", &AEDAT4::frames);
}