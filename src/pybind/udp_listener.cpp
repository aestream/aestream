#include <chrono>
#include <cstring>
#include <netdb.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

#include <torch/extension.h>
#include <torch/torch.h>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "udp_client.hpp"

class TensorBuffer {
private:
  const std::vector<int64_t> shape;
  const bool sum_events;
  torch::TensorOptions options_buffer;
  torch::TensorOptions options_copy;

  std::mutex buffer_lock;
  std::shared_ptr<torch::Tensor> buffer1;
  std::shared_ptr<torch::Tensor> buffer2;

public:
  TensorBuffer(torch::IntArrayRef size, std::string device, bool sum_events)
      : shape(size.vec()), sum_events(sum_events) {
    options_buffer = torch::TensorOptions()
                         .dtype(torch::kChar)
                         .device(torch::kCPU)
                         .memory_format(c10::MemoryFormat::Contiguous);
    options_copy = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    buffer1 =
        std::make_shared<torch::Tensor>(torch::zeros(size, options_buffer));
    buffer2 =
        std::make_shared<torch::Tensor>(torch::zeros(size, options_buffer));
  }
  void set_buffer(uint16_t data[], int numbytes) {
    const auto length = numbytes >> 1;
    buffer_lock.lock();
    char *array = (char *)buffer1->data_ptr();
    if (sum_events) {
      for (int i = 0; i < length; i = i + 2) {
        // Decode x, y
        const uint16_t y_coord = data[i] & 0x7FFF;
        const uint16_t x_coord = data[i + 1] & 0x7FFF;
        char value = (char)*(array + shape[1] * x_coord + y_coord);
        *(array + shape[1] * x_coord + y_coord) = value + 1;
      }
    } else {
      for (int i = 0; i < length; i = i + 2) {
        // Decode x, y
        const uint16_t y_coord = data[i] & 0x7FFF;
        const uint16_t x_coord = data[i + 1] & 0x7FFF;
        *(array + shape[1] * x_coord + y_coord) = 1;
      }
    }
    buffer_lock.unlock();
  }
  at::Tensor read() {
    // Swap out old pointer
    buffer_lock.lock();
    buffer1.swap(buffer2);
    buffer_lock.unlock();
    // Copy and clean
    auto copy = buffer2->to(options_copy, true, true);
    buffer2->index_put_({torch::indexing::Slice()}, 0);
    return copy;
  }
};

class UDPInput {
private:
  TensorBuffer buffer;
  const int port;
  const int max_events_per_packet = 1024;

  std::thread socket_thread;
  std::atomic<bool> is_serving = {true};

public:
  uint64_t count = 0;
  UDPInput(torch::IntArrayRef shape, std::string device, int port,
           bool sum_events)
      : buffer(shape, device, sum_events), port(port) {}

  UDPInput *start_server() {
    std::thread socket_thread(&UDPInput::serve_synchronous, this);
    socket_thread.detach();
    return this;
  }

  at::Tensor read() { return buffer.read(); }

  void serve_synchronous() {
    int sockfd;
    int numbytes;
    uint16_t int_buf[max_events_per_packet];

    // Connect to socket
    struct sockaddr_storage their_addr;
    socklen_t addr_len;
    sockfd = udp_client(std::to_string(port));
    addr_len = sizeof(their_addr);

    // start receiving event
    while (is_serving.load()) {
      if ((numbytes = recvfrom(sockfd, int_buf, sizeof(int_buf), 0,
                               (struct sockaddr *)&their_addr, &addr_len)) ==
          -1) {
        perror("recvfrom");
        return;
      }
      count += numbytes / 4;

      buffer.set_buffer(int_buf, numbytes);
    }
    close(sockfd);
  }

  void stop_server() { is_serving.store(false); }
};

PYBIND11_MODULE(aestream, m) {
  py::class_<UDPInput>(m, "UDPInput")
      .def(py::init<torch::IntArrayRef, std::string, int, bool>(),
           py::arg("shape"), py::arg("device") = "cpu", py::arg("port") = 3333,
           py::arg("sum_events") = false)
      .def("start", &UDPInput::start_server)
      .def("__enter__", &UDPInput::start_server)
      .def("__exit__",
           [&](UDPInput &i, py::object t, py::object v, py::object trace) {
             i.stop_server();
             return true;
           })
      .def("read", &UDPInput::read);
}