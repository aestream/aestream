#include "prophesee.hpp"

// event generator for Prophesee cameras
Generator<AER::Event> prophesee_event_generator(
    const std::atomic<bool> &runFlag,
    const std::optional<std::string> serial_number = std::nullopt) {

  Metavision::Camera cam;

  // get camera
  try {
    if (serial_number.has_value()) {
      cam = Metavision::Camera::from_serial(serial_number.value());
    } else {
      cam = Metavision::Camera::from_first_available();
    }
  } catch (const std::exception &e) {
    if (serial_number.has_value()) {
      std::cerr << "Failure with serial number '" << serial_number.value()
                << "': " << e.what() << std::endl;
    } else {
      std::cerr << "Failure to identify Prophesee camera: " << e.what()
                << std::endl;
    }

    Metavision::AvailableSourcesList available_systems =
        cam.list_online_sources();

    for (int i = 0;
         i < available_systems[Metavision::OnlineSourceType::USB].size(); i++) {
      std::cerr << "- "
                << available_systems[Metavision::OnlineSourceType::USB][i]
                << std::endl;
    }

    throw std::invalid_argument(
        "Please choose one of the above listed serial numbers and run again!");
  }

  const Metavision::EventCD *ev_start = NULL, *ev_final = NULL;

  // add event callback -> will set ev_start and ev_final to respective begin
  // and end of event buffer
  cam.cd().add_callback(
      [&ev_start, &ev_final](const Metavision::EventCD *ev_begin,
                             const Metavision::EventCD *ev_end) -> void {
        ev_start = ev_begin;
        ev_final = ev_end;
      });

  // start camera
  cam.start();

  // keep running while camera is on or video is finished
  while (cam.is_running() && runFlag.load()) {
    if ((ev_start != NULL) && (ev_final != NULL)) {
      // iterate over events in buffer and convert to AEDAT Polarity Event
      for (const Metavision::EventCD *ev = ev_start; ev < ev_final; ++ev) {
        const AER::Event polarityEvent = {
            (uint64_t)ev->t,
            ev->x,
            ev->y,
            (bool)ev->p,
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
