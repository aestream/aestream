#include "dvs_to_udp.hpp"

// Constructor - initialize socket
template <typename T>
DVSToUDP<T>::DVSToUDP(uint32_t bfsize, std::string port, std::string IP) {
  struct addrinfo hints, *servinfo;
  int rv;

  // Packet configs
  buffer_size = bfsize;

  // UDP configs
  serverport = port;
  IPAdress = IP;

  memset(&hints, 0, sizeof hints);
  hints.ai_family =
      AF_INET; // set to AF_INET to use IPv4, to AF_INET6 to use IPv6
  hints.ai_socktype = SOCK_DGRAM;

  if (IPAdress == "localhost")
    hints.ai_flags = AI_PASSIVE; // if IP adress not specified, use own IP

  if ((rv = getaddrinfo(IPAdress.c_str(), serverport.c_str(), &hints,
                        &servinfo)) != 0) {
    fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
    throw "Error raised";
  }

  // loop through all the results and make a socket
  for (p = servinfo; p != NULL; p = p->ai_next) {
    if ((sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1) {
      perror("talker: socket");
      continue;
    }

    break;
  }

  if (p == NULL) {
    fprintf(stderr, "talker: failed to create socket\n");
    throw "Error raised";
  }
}

// Process a packet of events and send it using UDP over the socket
template <typename T>
void DVSToUDP<T>::stream(Generator<T> &input_generator,
                         bool include_timestamp) {
  int numbytes;
  int event_size;
  uint64_t events_sent = 0;
  bool sent;
  int current_event = 0;

  if (include_timestamp) {
    event_size = 8;
  } else {
    event_size = 4;
  }

  const uint64_t max_events = UDP_max_bytesize / event_size;
  uint32_t message[max_events];
  uint64_t count = 0;

  for (AER::Event event : input_generator) {
    count += 1;
    sent = false;

    // Encoding according to protocol
    if (include_timestamp) {
      message[current_event] =
          (event.x & 0x7FFF)
          << 16; // Be aware that for machine-independance it should be:
                 // htons(polarity_event.x & 0x7FFF);
      message[current_event + 1] = event.timestamp;
    } else {
      message[current_event] =
          (event.x | 0x8000)
          << 16; // Be aware that for machine-independance it should be:
                 // htons(polarity_event.x | 0x8000);
    }

    if (event.polarity) {
      message[current_event] |=
          event.y | 0x8000; // Be aware that for machine-independance it
                            // should be: htons(polarity_event.y | 0x8000);
    } else {
      message[current_event] |=
          event.y & 0x7FFF; // Be aware that for machine-independance it
                            // should be: htons(polarity_event.y & 0x7FFF);
    }

    if (include_timestamp) {
      current_event += 2;
    } else {
      current_event += 1;
    }

    if (current_event == max_events) {
      if ((numbytes = sendto(sockfd, &message, sizeof(message), 0, p->ai_addr,
                             p->ai_addrlen)) == -1) {
        perror("talker error: sendto");
        exit(1);
      }

      sent = true;
      current_event = 0;
      events_sent += max_events;
    }
  }

  printf("Sent a total of %lu events\n", count);

  if (sent == false) {
    if ((numbytes = sendto(sockfd, &message, current_event * event_size, 0,
                           p->ai_addr, p->ai_addrlen)) == -1) {
      perror("talker error: sendto");
      exit(1);
    }
    events_sent += current_event;
  }
}

// Close the socket
template <typename T> void DVSToUDP<T>::closesocket() { close(sockfd); }

template class DVSToUDP<AER::Event>;