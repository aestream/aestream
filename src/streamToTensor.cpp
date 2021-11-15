#include <csignal>
#include <string>
#include <sys/types.h>
#include <stdexcept>

#include "CLI11.hpp"
#include "dvsToTensor.hpp"

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


    // Generate options
    //
    std::int64_t maxPackets = -1;
    app.add_option("--max-packets", maxPackets, "Maximum number of packets to read before stopping. Defaults to -1 (infinite).");

    CLI11_PARSE(app, argc, argv);

    //
    // Handle input
    //
    Generator<torch::Tensor> input_generator;
    if (app_input_inivation->parsed()) {
    input_generator = sparse_tensor_generator(camera, deviceId, deviceAddress);
    } else if (app_input_prophesee->parsed()) {
    input_generator = sparse_tensor_generator(serial_number);
    }

    //
    // Handle output
    //
    try {
    // Default to STDOUT
    for (torch::Tensor tensor : input_generator) {
        std::cout << "tensor processed" << std::endl;
    }
    } catch (const std::exception &e) {
    std::cout << "Failure while streaming events: " << e.what() << "\n";
    }
}