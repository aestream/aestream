#include <csignal>
#include <iostream>

#include "aedat.hpp"
#include "generator.hpp"
#include "usb.hpp"
#include "metavision/sdk/driver/camera.h"
#include "device_discovery.hpp"


USBConnection::USBConnection(std::string camera, uint16_t deviceId,
                             uint8_t deviceAddress) {

  try{                         
    if (camera == "dvx") {
      handle =
          new libcaer::devices::dvXplorer(deviceId, deviceId, deviceAddress, "");
    } else if (camera == "davis") {
      handle = new libcaer::devices::davis(deviceId, deviceId, deviceAddress, "");
    } else {
      throw std::invalid_argument("Unknown camera '" + camera + "'");
    }
  } catch (const std::exception &e) {
    std::cout << "Failure with camera: " << camera << ", deviceId: '" << deviceId << "', deviceAddress (devAddress) see below.\n" << std::endl;
    std::cout <<  e.what() << std::endl;

    std::cout << "List of available inivation cameras and configurations: " << std::endl;

    int exit_status = list_devices(); 

    std::cout << std::endl; 

    throw std::invalid_argument("Please choose the deviceId (busNum) and deviceAddress (devAddr) of one of the above listed iniVation cameras!");
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

// event generator for Prophesee cameras
Generator<AEDAT::PolarityEvent> 
usb_event_generator(const std::string serial_number = "None"){

    Metavision::Camera cam; // = Metavision::Camera::from_first_available(); 

    
    Metavision::AvailableSourcesList available_systems = cam.list_online_sources(); 

    // get camera by serial number available camera
    try{
      cam = Metavision::Camera::from_serial(serial_number);
    } catch (const std::exception &e) {
      std::cout << "Failure with serial number '" << serial_number << "': " <<e.what() << std::endl;

      std::cout << "Serial numbers of available cameras: " << std::endl; 

      for (int i = 0; i < available_systems[Metavision::OnlineSourceType::USB].size(); i++){
        std::cout << "- " << available_systems[Metavision::OnlineSourceType::USB][i] << std::endl;
      }

      std::cout << std::endl; 

      throw std::invalid_argument("Please choose one of the above listed serial numbers and run again!"); 
    }
    
    const Metavision::EventCD *ev_start = NULL, *ev_final = NULL; 

    // add event callback -> will set ev_start and ev_final to respective begin and end of event buffer
    cam.cd().add_callback([&ev_start, &ev_final](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) -> void{
        ev_start = ev_begin; 
        ev_final = ev_end; 
    });

    // start camera
    cam.start();

    // keep running while camera is on or video is finished
    while (cam.is_running()){
        if ((ev_start != NULL) && (ev_final != NULL)){
            // iterate over events in buffer and convert to AEDAT Polarity Event
            for (const Metavision::EventCD *ev = ev_start; ev < ev_final; ++ev){
                const AEDAT::PolarityEvent polarityEvent = {
                    true,
                    ev->p,
                    ev->x,
                    ev->y,
                    (uint64_t)ev->t,
                };
                co_yield polarityEvent;
            }
            ev_start = NULL; 
            ev_final = NULL; 
        }
    }

    // if video is finished, stop camera - will never get here with live camera
    cam.stop();
} 
