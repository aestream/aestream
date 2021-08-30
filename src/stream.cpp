#include <csignal>
#include <string>
#include <sys/types.h>
#include <stdexcept>

#include "CLI11.hpp"

#include "aedat.hpp"
#include "aedat4.hpp"
#include "usb.hpp"
#include "dvsToUDP.hpp"

int main(int argc, char *argv[]) {
  CLI::App app{"Streams DVS data from a USB camera or AEDAT file to a file or "
               "UDP socket"};
  
  //
  // Input
  //
  auto app_input = app.add_subcommand("input", "Input source. Required")->required();

  // - DVS
  std::uint16_t deviceId;
  std::uint16_t deviceAddress;
  std::string camera = "davis";
  std::string serial_number;
  // Inivation cameras
  auto app_input_inivation = app_input->add_subcommand("inivation", "DVS input source for inivation cameras");
  app_input_inivation->add_option("id", deviceId, "Hardware ID")->required();
  app_input_inivation->add_option("address", deviceAddress, "Hardware address")->required();
  app_input_inivation->add_option("camera", camera, "Type of camera; davis or dvx")->required();
  // Prophesee cameras 
  auto app_input_prophesee = app_input->add_subcommand("prophesee", "DVS input source for prophesee cameras");
  app_input_prophesee->add_option("serial", serial_number, "Serial number")->required();
  // - File
  std::string filename = "None";
  auto app_input_file = app_input->add_subcommand("file", "AEDAT4 input file");
  app_input_file->add_option("file", filename, "Path to .aedat file")->required();

  //
  // Output
  //
  auto app_output = app.add_subcommand("output", "Output target. Defaults to stdout");
  // - SPIF
  std::string port = "3333";           // Port number
  std::string ipAddress = "localhost"; // IP Adress - if NULL, use own IP.
  std::uint32_t bufferSize = 1024;
  std::uint16_t packetSize = 128;
  bool include_timestamp = false; 
  auto app_output_spif = app_output->add_subcommand("spif", "SpiNNaker Interface Board (SPIF) output");
  app_output_spif->add_option("destination", ipAddress, "Destination IP. Defaults to localhost");
  app_output_spif->add_option("port", port, "Destination port. Defaults to 3333");
  app_output_spif->add_option("--buffer-size", bufferSize, "UDP buffer size. Defaults to 1024");
  app_output_spif->add_option("--packet-size", packetSize,  "Number of events in a single UDP packet. Defaults to 128");
  app_output_spif->add_option("--include-timestamp", include_timestamp,  "Include timestamp in events");

  //
  // Generate options
  //
  std::int64_t maxPackets = -1;
  app.add_option("--max-packets", maxPackets,
                 "Maximum number of packets to read before stopping. Defaults to -1 (infinite).");

  CLI11_PARSE(app, argc, argv);

  //
  // Handle input
  //
  Generator<AEDAT::PolarityEvent> input_generator;
  if (app_input_inivation->parsed()) {
    input_generator = usb_event_generator(camera, deviceId, deviceAddress);
  } else if (app_input_prophesee->parsed()) {
    input_generator = usb_event_generator(serial_number);
  } else if (app_input_file->parsed()) {
    input_generator = file_event_generator(filename);
  }

  //
  // Handle output
  //
  try {
    if (app_output_spif->parsed()) {
      std::cout << "Send events to: " << ipAddress << " on port: " << port <<std::endl;
      DVSToUDP<AEDAT::PolarityEvent> client(packetSize, bufferSize, port, ipAddress);
      client.sendpacket(input_generator, include_timestamp);
    } else { // Default to STDOUT
      for (AEDAT::PolarityEvent event : input_generator) {
        std::cout << event.x << ", " << event.y << ", " << event.timestamp << ";" << std::endl;
      }
    }
  } catch (const std::exception &e) {
    std::cout << "Failure while streaming events: " << e.what() << "\n";
  }
}