# include "dvsToTensor.hpp"

// sparse tensor generator for Inivation cameras
Generator<torch::Tensor>
sparse_tensor_generator(std::string camera, std::uint16_t deviceId, std::uint8_t deviceAddress) {

  auto connection = USBConnection(camera, deviceId, deviceAddress);

  std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer;
  std::vector<AEDAT::PolarityEvent> polarity_events;

  while (true) {
    do {
      packetContainer = connection.getPacket();
    } while (packetContainer == nullptr);

    for (auto &packet : *packetContainer) {
        if (packet == nullptr) {
            continue; // Skip if nothing there.
        }

        if(packet->getEventType() == POLARITY_EVENT){  
            std::shared_ptr<const libcaer::events::PolarityEventPacket> polarity = std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);

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

                polarity_events.push_back(polarityEvent);
            }
            auto event_tensors = convert_polarity_events(polarity_events);
            std::vector<AEDAT::PolarityEvent>().swap(polarity_events);
            co_yield event_tensors;
        }
    }
  }
};


// sparse tensor generator for Prophesee cameras
Generator<torch::Tensor>
sparse_tensor_generator(const std::string serial_number = "None"){

    Metavision::Camera cam; // = Metavision::Camera::from_first_available(); 

    std::vector<AEDAT::PolarityEvent> polarity_events;
    
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

                polarity_events.push_back(polarityEvent);

            }
            ev_start = NULL; 
            ev_final = NULL; 

            auto event_tensors = convert_polarity_events(polarity_events);
            std::vector<AEDAT::PolarityEvent>().swap(polarity_events);
            co_yield event_tensors;
        }
    }

    // if video is finished, stop camera - will never get here with live camera
    cam.stop();
} 