#pragma once

#include <string>

// socket programming
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "../aer.hpp"
#include "../generator.hpp"

template <typename T> class DVSToUDP {
public:
  int sockfd = -1;
  std::string serverport;
  std::string IPAdress;
  struct addrinfo *p = NULL;
  uint32_t buffer_size;

  static const uint16_t UDP_max_bytesize = 512;

  DVSToUDP(uint32_t bfsize, std::string port, std::string IP);

  void stream(Generator<T> &input_generator, bool include_timestamp);
  void closesocket();
};