#pragma once

#include <any>
#include <csignal>
#include <iostream>
#include <string>

#include <libcaercpp/devices/davis.hpp>
#include <libcaercpp/devices/dvxplorer.hpp>

#include "../aedat.hpp"
#include "../generator.hpp"

class CAERUSBConnection {
  uint32_t containerInterval = 128;
  uint32_t bufferSize = 1024;
  libcaer::devices::device *handle;

  //   static void signalHandler(int signal) { close(); }
  static void shutdownHandler(void *ptr) {
    // Unused
  }

public:
  CAERUSBConnection(std::string camera, std::uint16_t deviceId,
                    std::uint8_t deviceAddress);
  ~CAERUSBConnection() { close(); }

  std::unique_ptr<libcaer::events::EventPacketContainer> getPacket() {
    return handle->dataGet();
  }
  void close() { handle->dataStop(); }
};

Generator<AEDAT::PolarityEvent>
inivation_event_generator(std::string camera, std::uint16_t deviceId,
                          std::uint8_t deviceAddress);