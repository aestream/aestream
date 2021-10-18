#pragma once

#include <any>
#include <csignal>
# include <string>

#include <libcaercpp/devices/davis.hpp>
#include <libcaercpp/devices/dvxplorer.hpp>

# include <metavision/sdk/driver/camera.h>
# include <metavision/sdk/base/events/event_cd.h>

#include "aedat.hpp"
#include "generator.hpp"

class USBConnection {
  uint32_t containerInterval = 128;
  uint32_t bufferSize = 1024;
  libcaer::devices::device *handle;

  //   static void signalHandler(int signal) { close(); }
  static void shutdownHandler(void *ptr) {
    // Unused
  }

public:
  USBConnection(std::string camera, std::uint16_t deviceId,
                std::uint8_t deviceAddress);
  ~USBConnection() { close(); }

  std::unique_ptr<libcaer::events::EventPacketContainer> getPacket() {
    return handle->dataGet();
  }
  void close() { handle->dataStop(); }
};

Generator<AEDAT::PolarityEvent>
usb_event_generator(std::string camera, std::uint16_t deviceId,
                    std::uint8_t deviceAddress);
