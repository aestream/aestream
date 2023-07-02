#include <atomic>
#include <zmq.hpp>
#include <iostream>

#include "input/zmq.hpp"

Generator<AER::Event> open_zmq(const std::string socket, std::atomic<bool>& runFlag) {
  zmq::context_t ctx;
  zmq::socket_t sock(ctx, zmq::socket_type::pair);
  try {
    sock.bind(socket);
  } catch (std::exception e) {
    std::cout << e.what() << std::endl;
  }

  zmq::message_t message;
  while (runFlag.load()) {
    std::cout << "Receiving " << std::endl;
    sock.recv(message);
    char *data = message.data<char>();
    std::cout << data << std::endl;
    co_yield {0, 0, 0, 0};
  }
  sock.close();
}