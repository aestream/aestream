#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

#include "types.hpp"

#include "tensor_buffer.hpp"
#include "udp_client.hpp"

class UDPInput {
private:
  TensorBuffer buffer;
  const int port;
  static const int max_events_per_packet = 256;

  std::thread socket_thread;
  std::atomic<bool> is_serving = {true};

public:
  uint64_t count = 0;
  UDPInput(py_size_t shape, const std::string &device, int port)
      : buffer(shape, device, max_events_per_packet), port(port) {}

  UDPInput *start_stream() {
    std::thread socket_thread(&UDPInput::serve_synchronous, this);
    socket_thread.detach();
    return this;
  }

  std::unique_ptr<BufferPointer> read() { return buffer.read(); }
  void read_genn(uint32_t *bitmask, size_t size){ buffer.read_genn(bitmask, size); }
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

  void stop_stream(nb::object &a, nb::object &b, nb::object &c) {
    is_serving.store(false);
  }
};
