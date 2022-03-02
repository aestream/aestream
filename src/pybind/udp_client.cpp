#include <cstring>
#include <string>

#include "./udp_client.hpp"

int udp_client(std::string port) {
  // socket variables
  int sockfd;
  struct addrinfo hints, *servinfo, *p;
  int rv;

  // establish connection for client
  memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_INET; // set to AF_INET to use IPv4
  hints.ai_socktype = SOCK_DGRAM;
  hints.ai_flags = AI_PASSIVE; // use my IP

  const std::string ip = "0.0.0.0";

  // Get adrress-info
  if ((rv = getaddrinfo(ip.c_str(), port.c_str(), &hints, &servinfo)) != 0) {
    fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
    return 1;
  }

  // loop through all the results and bind to the first we can
  for (p = servinfo; p != NULL; p = p->ai_next) {
    if ((sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1) {
      perror("listener: socket");
      continue;
    }

    if (bind(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
      close(sockfd);
      perror("listener: bind");
      continue;
    }
    break;
  }

  if (p == NULL) {
    fprintf(stderr, "listener: failed to bind socket\n");
    return 2;
  }

  freeaddrinfo(servinfo);

  return sockfd;
}