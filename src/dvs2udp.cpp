#define LIBCAER_FRAMECPP_OPENCV_INSTALLED 0

#include "dvs2udp.hpp"
#include <libcaer/devices/davis.h>
#include <libcaercpp/devices/davis.hpp>
#include <libcaercpp/devices/dvxplorer.hpp>

#include <atomic>
#include <csignal>

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

// Constructor - initialize socket
template <class cameratype>
DVSStream<cameratype>::DVSStream(uint32_t interval, uint32_t bfsize,
                                 std::string port, std::string IP,
                                 struct addrinfo *point, std::string file) {
  struct addrinfo hints, *servinfo;
  int rv;

  // Packet configs
  container_interval = interval;
  buffer_size = bfsize;

  // UDP configs
  serverport = port;
  IPAdress = IP;
  p = point;

  // filename = file;

  // Open file for writing events if specified
  // if (filename == "None") {
  //   printf("Write output to file: %s\n", filename);
  //   fileOutput.open(filename, std::fstream::app);
  // }

  memset(&hints, 0, sizeof hints);
  hints.ai_family =
      AF_INET; // set to AF_INET to use IPv4, to AF_INET6 to use IPv6
  hints.ai_socktype = SOCK_DGRAM;

  if (IPAdress == "") {
    hints.ai_flags = AI_PASSIVE; // if IP adress not specified, use my IP
  }

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

template class DVSStream<libcaer::devices::davis>;
template class DVSStream<libcaer::devices::dvXplorer>;

// Global Shutdown Handler
template <class cameratype>
void DVSStream<cameratype>::globalShutdownSignalHandler(int signal) {
  static atomic_bool globalShutdown(false);
  // Simply set the running flag to false on SIGTERM and SIGINT (CTRL+C) for
  // global shutdown.
  if (signal == SIGTERM || signal == SIGINT) {
    globalShutdown.store(true);
  }
}

// USB Shutdown Handler
template <class cameratype>
void DVSStream<cameratype>::usbShutdownHandler(void *ptr) {
  static atomic_bool globalShutdown(false);
  (void)(ptr); // UNUSED.

  globalShutdown.store(true);
}

// Open a DAVIS given a USB ID, and don't care about USB bus or SN restrictions.
template <class cameratype>
libcaer::devices::davis DVSStream<cameratype>::connect2davis(int ID,
                                                             int devAddress) {

#if defined(_WIN32)
  if (signal(SIGTERM, &globalShutdownSignalHandler) == SIG_ERR) {
    libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
                      "Failed to set signal handler for SIGTERM. Error: %d.",
                      errno);
    return (EXIT_FAILURE);
  }

  if (signal(SIGINT, &globalShutdownSignalHandler) == SIG_ERR) {
    libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
                      "Failed to set signal handler for SIGINT. Error: %d.",
                      errno);
    return (EXIT_FAILURE);
  }
#else
  struct sigaction shutdownAction;

  shutdownAction.sa_handler = &DVSStream::globalShutdownSignalHandler;
  shutdownAction.sa_flags = 0;
  sigemptyset(&shutdownAction.sa_mask);
  sigaddset(&shutdownAction.sa_mask, SIGTERM);
  sigaddset(&shutdownAction.sa_mask, SIGINT);

  if (sigaction(SIGTERM, &shutdownAction, NULL) == -1) {
    libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
                      "Failed to set signal handler for SIGTERM. Error: %d.",
                      errno);
    return (EXIT_FAILURE);
  }

  if (sigaction(SIGINT, &shutdownAction, NULL) == -1) {
    libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
                      "Failed to set signal handler for SIGINT. Error: %d.",
                      errno);
    return (EXIT_FAILURE);
  }
#endif

  libcaer::devices::davis davisHandle =
      libcaer::devices::davis(ID, ID, devAddress, "");

  // Let's take a look at the information we have on the device.
  struct caer_davis_info davis_info = davisHandle.infoGet();

  printf("Streaming from %s --- ID: %d, Master: %d, DVS X: %d, DVS Y: %d, "
         "Logic: %d.\n",
         davis_info.deviceString, davis_info.deviceID,
         davis_info.deviceIsMaster, davis_info.dvsSizeX, davis_info.dvsSizeY,
         davis_info.logicVersion);

  // Send the default configuration before using the device.
  // No configuration is sent automatically!
  davisHandle.sendDefaultConfig();

  // Tweak some biases, to increase bandwidth in this case.
  struct caer_bias_coarsefine coarseFineBias;

  coarseFineBias.coarseValue = 2;
  coarseFineBias.fineValue = 116;
  coarseFineBias.enabled = true;
  coarseFineBias.sexN = false;
  coarseFineBias.typeNormal = true;
  coarseFineBias.currentLevelNormal = true;

  davisHandle.configSet(DAVIS_CONFIG_BIAS, DAVIS240_CONFIG_BIAS_PRBP,
                        caerBiasCoarseFineGenerate(coarseFineBias));

  coarseFineBias.coarseValue = 1;
  coarseFineBias.fineValue = 33;
  coarseFineBias.enabled = true;
  coarseFineBias.sexN = false;
  coarseFineBias.typeNormal = true;
  coarseFineBias.currentLevelNormal = true;

  davisHandle.configSet(DAVIS_CONFIG_BIAS, DAVIS240_CONFIG_BIAS_PRSFBP,
                        caerBiasCoarseFineGenerate(coarseFineBias));

  // Let's verify they really changed!
  uint32_t prBias =
      davisHandle.configGet(DAVIS_CONFIG_BIAS, DAVIS240_CONFIG_BIAS_PRBP);
  uint32_t prsfBias =
      davisHandle.configGet(DAVIS_CONFIG_BIAS, DAVIS240_CONFIG_BIAS_PRSFBP);

  printf("New bias values --- PR-coarse: %d, PR-fine: %d, PRSF-coarse: %d, "
         "PRSF-fine: %d.\n",
         caerBiasCoarseFineParse(prBias).coarseValue,
         caerBiasCoarseFineParse(prBias).fineValue,
         caerBiasCoarseFineParse(prsfBias).coarseValue,
         caerBiasCoarseFineParse(prsfBias).fineValue);

  return davisHandle;
}

template <class cameratype>
libcaer::devices::dvXplorer DVSStream<cameratype>::connect2dvx(int ID,
                                                               int devAddress) {
  // Install signal handler for global shutdown.
  // #if defined(_WIN32)
  //   if (signal(SIGTERM, &globalShutdownSignalHandler) == SIG_ERR) {
  //     libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
  //                       "Failed to set signal handler for SIGTERM. Error:
  //                       %d.", errno);
  //     return (EXIT_FAILURE);
  //   }

  //   if (signal(SIGINT, &globalShutdownSignalHandler) == SIG_ERR) {
  //     libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
  //                       "Failed to set signal handler for SIGINT. Error:
  //                       %d.", errno);
  //     return (EXIT_FAILURE);
  //   }
  // #else
  //   struct sigaction shutdownAction;

  //   shutdownAction.sa_handler = &globalShutdownSignalHandler;
  //   shutdownAction.sa_flags = 0;
  //   sigemptyset(&shutdownAction.sa_mask);
  //   sigaddset(&shutdownAction.sa_mask, SIGTERM);
  //   sigaddset(&shutdownAction.sa_mask, SIGINT);

  //   if (sigaction(SIGTERM, &shutdownAction, NULL) == -1) {
  //     libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
  //                       "Failed to set signal handler for SIGTERM. Error:
  //                       %d.", errno);
  //     return (EXIT_FAILURE);
  //   }

  //   if (sigaction(SIGINT, &shutdownAction, NULL) == -1) {
  //     libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
  //                       "Failed to set signal handler for SIGINT. Error:
  //                       %d.", errno);
  //     return (EXIT_FAILURE);
  //   }
  // #endif

  // Open a DVS, give it a device ID of 1, and don't care about USB bus or SN
  // restrictions.
  auto handle = libcaer::devices::dvXplorer(ID, ID, devAddress, "");

  // Let's take a look at the information we have on the device.
  auto info = handle.infoGet();

  printf("%s --- ID: %d, DVS X: %d, DVS Y: %d, Firmware: %d, Logic: %d.\n",
         info.deviceString, info.deviceID, info.dvsSizeX, info.dvsSizeY,
         info.firmwareVersion, info.logicVersion);

  // Send the default configuration before using the device.
  // No configuration is sent automatically!
  handle.sendDefaultConfig();

  return handle;
}

// Start getting some data from the device. We just loop in blocking mode,
// no notification needed regarding new events. The shutdown notification, for
// example if the device is disconnected, should be listened to.
template <class cameratype>
cameratype DVSStream<cameratype>::startdatastream(cameratype davisHandle) {

  // Set parsing intervall where container interval is in [10μs] unit
  // davisHandle.configSet(CAER_HOST_CONFIG_PACKETS,
  // CAER_HOST_CONFIG_PACKETS_MAX_CONTAINER_INTERVAL, container_interval);

  // Set number of events per packet
  davisHandle.configSet(CAER_HOST_CONFIG_PACKETS,
                        CAER_HOST_CONFIG_PACKETS_MAX_CONTAINER_PACKET_SIZE,
                        container_interval);

  // Configs about buffer
  davisHandle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE,
                        CAER_HOST_CONFIG_DATAEXCHANGE_BUFFER_SIZE, buffer_size);

  uint32_t BFSize = davisHandle.configGet(
      CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BUFFER_SIZE);
  printf("Buffer size: %d\n", BFSize);

  // Start data stream
  davisHandle.dataStart(nullptr, nullptr, nullptr,
                        &DVSStream::usbShutdownHandler, nullptr);

  // Let's turn on blocking data-get mode to avoid wasting resources.
  davisHandle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE,
                        CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);

  return davisHandle;
}

// Process a packet of events and send it using UDP over the socket
template <class cameratype>
void DVSStream<cameratype>::sendpacket(cameratype davisHandle,
                                       bool include_timestamp) {
  std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer =
      nullptr;
  int numbytes;
  int event_size;
  uint16_t max_events;
  bool sent;
  int current_event = 0;

  if (include_timestamp) {
    event_size = 8;
  } else {
    event_size = 4;
  }

  max_events = UDP_max_bytesize / event_size;

  do {
    packetContainer = davisHandle.dataGet();
  } while (packetContainer == nullptr);

  for (auto &packet : *packetContainer) {
    if (packet == nullptr) {
      continue; // Skip if nothing there.
    }

    if (packet->getEventType() == POLARITY_EVENT) {
      std::shared_ptr<const libcaer::events::PolarityEventPacket> polarity =
          std::static_pointer_cast<libcaer::events::PolarityEventPacket>(
              packet);

      for (const auto &evt : *polarity) {
        if (evt.isValid() == true) {
          DVSevents::PolarityEvent polarity_event;
          sent = false;

          polarity_event.timestamp = evt.getTimestamp64(*polarity);
          polarity_event.x = evt.getX();
          polarity_event.y = evt.getY();
          polarity_event.polarity = evt.getPolarity();
          if (evt.getX() > 639 || evt.getY() > 479 || evt.getX() < 0 || evt.getY() < 0) {
            printf("%dx%d ", evt.getX(), evt.getY());
          }

          // If specified write to file
          // if (filename == "None") {
          //   fileOutput << "DVS " << polarity_event.timestamp << " "
          //              << polarity_event.x << " " << polarity_event.y << " "
          //              << polarity_event.polarity << std::endl;
          // }

          // Encoding according to protocol
          if (include_timestamp) {
            message[current_event] =
                (polarity_event.x & 0x7FFF)
                << 16; // Be aware that for machine-independance it should be:
                       // htons(polarity_event.x & 0x7FFF);

            message[current_event + 1] = evt.getTimestamp();
          } else {
            message[current_event] =
                (polarity_event.x | 0x8000)
                << 16; // Be aware that for machine-independance it should be:
                       // htons(polarity_event.x | 0x8000);
          }

          if (polarity_event.polarity) {
            message[current_event] |=
                polarity_event.y |
                0x8000; // Be aware that for machine-independance it should be:
                        // htons(polarity_event.y | 0x8000);
          } else {
            message[current_event] |=
                polarity_event.y &
                0x7FFF; // Be aware that for machine-independance it should be:
                        // htons(polarity_event.y & 0x7FFF);
          }

          if (include_timestamp) {
            current_event += 2;
          } else {
            current_event += 1;
          }
        }

        if (current_event == max_events) {
          if ((numbytes = sendto(DVSStream::sockfd, &message, sizeof(message),
                                 0, p->ai_addr, p->ai_addrlen)) == -1) {
            perror("talker error: sendto");
            exit(1);
          }

          sent = true;
          current_event = 0;
          events_sent += max_events;
        }
      }

      // if (strcmp(filename, "None") != 0) {
      //   fileOutput << "=================" << std::endl;
      // }
    }
  }

  if (sent == false) {
    if ((numbytes =
             sendto(DVSStream::sockfd, &message, current_event * event_size, 0,
                    p->ai_addr, p->ai_addrlen)) == -1) {
      perror("talker error: sendto");
      exit(1);
    }
    events_sent += current_event;
  }
}

// Stops the datastream
template <class cameratype>
int DVSStream<cameratype>::stopdatastream(cameratype davisHandle) {
  davisHandle.dataStop();
  // Close automatically done by destructor.
  printf("Shutdown successful.\n");
  return (EXIT_FAILURE);
}

// Close the socket
template <class cameratype> void DVSStream<cameratype>::closesocket() {
  close(sockfd);
}
