#include <csignal>

#include "aedat.hpp"
#include "generator.hpp"
#include "usb.hpp"

USBConnection::USBConnection(std::string camera, uint16_t deviceId,
                             uint8_t deviceAddress) {

  if (camera == "dvx") {
    handle =
        new libcaer::devices::dvXplorer(deviceId, deviceId, deviceAddress, "");
  } else if (camera == "davis") {
    handle = new libcaer::devices::davis(deviceId, deviceId, deviceAddress, "");
  } else {
    throw std::invalid_argument("Unknown camera '" + camera + "'");
  }

  // Send the default configuration before using the device.
  // No configuration is sent automatically!
  handle->sendDefaultConfig();

  // Set parsing intervall where container interval is in [10μs] unit
  // davisHandle.configSet(CAER_HOST_CONFIG_PACKETS,
  // CAER_HOST_CONFIG_PACKETS_MAX_CONTAINER_INTERVAL, container_interval);

  // Set number of events per packet
  handle->configSet(CAER_HOST_CONFIG_PACKETS,
                    CAER_HOST_CONFIG_PACKETS_MAX_CONTAINER_PACKET_SIZE,
                    containerInterval);

  // Configs about buffer
  handle->configSet(CAER_HOST_CONFIG_DATAEXCHANGE,
                    CAER_HOST_CONFIG_DATAEXCHANGE_BUFFER_SIZE, bufferSize);

  // Start data stream
  handle->dataStart(nullptr, nullptr, nullptr, &shutdownHandler, nullptr);

  // Let's turn on blocking data-get mode to avoid wasting resources.
  handle->configSet(CAER_HOST_CONFIG_DATAEXCHANGE,
                    CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);
}

Generator<AEDAT::PolarityEvent>
usb_event_generator(std::string camera, std::uint16_t deviceId,
                    std::uint8_t deviceAddress) {

  auto connection = USBConnection(camera, deviceId, deviceAddress);

  std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer;
  while (true) {
    do {
      packetContainer = connection.getPacket();
    } while (packetContainer == nullptr);

    for (auto &packet : *packetContainer) {
      if (packet == nullptr) {
        continue; // Skip if nothing there.
      }

      std::shared_ptr<const libcaer::events::PolarityEventPacket> polarity =
          std::static_pointer_cast<libcaer::events::PolarityEventPacket>(
              packet);

      for (const libcaer::events::PolarityEvent &evt : *polarity) {
        if (!evt.isValid()) {
          continue;
        }

        const AEDAT::PolarityEvent polarityEvent = {
            evt.isValid(),
            evt.getPolarity(),
            evt.getX(),
            evt.getY(),
            (uint64_t)evt.getTimestamp64(*polarity),
        };

        co_yield polarityEvent;
      }
    }
  }
};