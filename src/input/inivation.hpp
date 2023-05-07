#pragma once

#include <any>
#include <atomic>
#include <csignal>
#include <iostream>
#include <string>

#include <libcaer/devices/device.h>
#include <libcaer/devices/device_discover.h>
#include <libcaercpp/devices/davis.hpp>
#include <libcaercpp/devices/dvxplorer.hpp>

#include "../aer.hpp"
#include "../generator.hpp"

struct InivationDeviceAddress {
  const std::string camera;
  const std::uint16_t deviceId;
  const std::uint16_t deviceAddress;
};

std::optional<libcaer::devices::device *> find_device();

class CAERUSBConnection {
  uint32_t containerInterval = 128;
  uint32_t bufferSize = 1024;
  libcaer::devices::device *handle;

  //   static void signalHandler(int signal) { close(); }
  static void shutdownHandler(void *ptr) {
    // close();
  }

public:
  CAERUSBConnection(std::optional<InivationDeviceAddress> deviceAddress);
  ~CAERUSBConnection() { close(); }

  std::unique_ptr<libcaer::events::EventPacketContainer> getPacket() {
    return handle->dataGet();
  }
  void close() { 
    try {
      handle->dataStop();
    } catch (const std::runtime_error &e) {
      // Ignore exceptions
    }
  }
};

Generator<AER::Event>
inivation_event_generator(std::optional<InivationDeviceAddress> device_address,
                          const std::atomic<bool> &runFlag);