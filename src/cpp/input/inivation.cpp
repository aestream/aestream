#include "./inivation.hpp"

std::optional<libcaer::devices::device *> find_device() {
  caerDeviceDiscoveryResult discovered;
  ssize_t result = caerDeviceDiscover(CAER_DEVICE_DISCOVER_ALL, &discovered);

  if (result < 0) {
    return {};
  } else {
    const auto device = discovered[0];
    switch (device.deviceType) {
    case CAER_DEVICE_DAVIS:

    case CAER_DEVICE_DVS128:
      return new libcaer::devices::davis(1);

    case CAER_DEVICE_DVXPLORER:
      return new libcaer::devices::dvXplorer(1);

      // case CAER_DEVICE_SAMSUNG_EVK:
      // case CAER_DEVICE_DAVIS_FX2:
      // case CAER_DEVICE_DAVIS_FX3:
      // case CAER_DEVICE_DAVIS_RPI:
      // case CAER_DEVICE_DYNAPSE:

    default:
      throw std::runtime_error("Cannot connect to unknown device of type " +
                               std::to_string(device.deviceType));
    };
  }
}

CAERUSBConnection::CAERUSBConnection(
    std::optional<InivationDeviceAddress> deviceAddress) {

  if (deviceAddress.has_value()) {
    const auto &[camera, deviceId, deviceHardwareAddress] =
        deviceAddress.value();
    if (camera == "dvx") {
      handle = new libcaer::devices::dvXplorer(deviceId, deviceId,
                                               deviceHardwareAddress, "");
    } else if (camera == "davis") {
      handle = new libcaer::devices::davis(deviceId, deviceId,
                                           deviceHardwareAddress, "");
    } else {
      throw std::invalid_argument("Unsupported camera '" + camera + "'");
    }
  } else {
    std::optional<libcaer::devices::device *> found_device = find_device();
    if (found_device.has_value()) {
      handle = found_device.value();
    } else {
      throw std::invalid_argument("No inivation device found.");
    }
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

// event generator for Inivation cameras
Generator<AER::Event>
inivation_event_generator(std::optional<InivationDeviceAddress> device_address,
                          const std::atomic<bool> &runFlag) {

  auto connection = CAERUSBConnection(device_address);

  std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer;
  try {
    while (runFlag.load()) {
      do {
        packetContainer = connection.getPacket();
      } while (packetContainer == nullptr);

      for (auto &packet : *packetContainer) {
        if (packet == nullptr) {
          continue; // Skip if nothing there.
        }

        if (packet->getEventType() == POLARITY_EVENT) {
          std::shared_ptr<const libcaer::events::PolarityEventPacket> polarity =
              std::static_pointer_cast<libcaer::events::PolarityEventPacket>(
                  packet);

          for (const libcaer::events::PolarityEvent &evt : *polarity) {
            if (!evt.isValid()) {
              continue;
            }

            const AER::Event polarityEvent = {
                (uint64_t)evt.getTimestamp64(*polarity),
                evt.getX(),
                evt.getY(),
                evt.getPolarity(),
            };

            co_yield polarityEvent;
          }
        }
      }
    }
  } catch (std::runtime_error &e) {
    std::cout << "Stream ending: " << e.what() << std::endl;
  }
};
