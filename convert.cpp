#include "convert.hpp"
#include "dvs_gesture.hpp"

#include <torch/csrc/autograd/python_variable.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

torch::Tensor convert_polarity_events_cropped(
    std::vector<AEDAT::PolarityEvent> &polarity_events,
    const size_t max_events) {
  const size_t size = max_events <= 0 ? polarity_events.size() : max_events;
  std::vector<int64_t> indices(3 * size);
  std::vector<int8_t> values;

  for (size_t idx = 0; idx < size; idx++) {
    auto event = polarity_events[idx];
    indices[idx] = event.timestamp - polarity_events[0].timestamp;
    indices[size + idx] = event.x;
    indices[2 * size + idx] = event.y;
    values.push_back(event.polarity ? 1 : -1);
  }

  auto index_options = torch::TensorOptions().dtype(torch::kInt64);
  torch::Tensor ind = torch::from_blob(
      indices.data(), {3, static_cast<uint32_t>(size)}, index_options);

  auto value_options = torch::TensorOptions().dtype(torch::kInt8);
  torch::Tensor val = torch::from_blob(
      values.data(), {static_cast<uint32_t>(values.size())}, value_options);
  auto events = torch::sparse_coo_tensor(ind, val);

  return events.clone();
}

torch::Tensor
convert_polarity_events(std::vector<AEDAT::PolarityEvent> &polarity_events) {
  return convert_polarity_events_cropped(polarity_events,
                                         polarity_events.size());
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
        "Converts the AEDAT data into a sparse Torch tensor");
  m.def("convert_polarity_events_cropped", &convert_polarity_events_cropped,
        "Converts the AEDAT data into a sparse Torch tensor, cropped to a "
        "maximum number of events");

  py::class_<AEDAT4>(m, "AEDAT4")
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def("load", &AEDAT4::load)
      .def_readwrite("polarity_events", &AEDAT4::polarity_events)
      .def_readwrite("frames", &AEDAT4::frames);
}