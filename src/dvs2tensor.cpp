#define LIBCAER_FRAMECPP_OPENCV_INSTALLED 0

#include <libcaercpp/devices/davis.hpp>
#include <libcaer/devices/davis.h>
#include "dvs2tensor.hpp"

#include <cstddef>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <type_traits>

// ================================
#include <atomic>
#include <csignal>

//=================================
// include guard
#ifndef PYBIND11_H
#define PYBIND11_H
#include <pybind11/pybind11.h>
#endif

#ifndef STL_H
#define STL_H
#include <pybind11/stl.h>
#endif

namespace py = pybind11;


// Constructor 
DVSDataConv::DVSDataConv(uint32_t interval, uint32_t bfsize){
    container_interval = interval; 
    buffer_size = bfsize;
}

void DVSDataConv::globalShutdownSignalHandler(int signal) {
    static atomic_bool globalShutdown(false);
    // Simply set the running flag to false on SIGTERM and SIGINT (CTRL+C) for global shutdown.
    if (signal == SIGTERM || signal == SIGINT) {
        globalShutdown.store(true);
    }
}

void DVSDataConv::usbShutdownHandler(void *ptr) {
    static atomic_bool globalShutdown(false);
	(void) (ptr); // UNUSED.

	globalShutdown.store(true);
}

// Open a DAVIS, given ID, and don't care about USB bus or SN restrictions.
void DVSDataConv::connect2camera(int ID){

    #if defined(_WIN32)
        if (signal(SIGTERM, &globalShutdownSignalHandler) == SIG_ERR) {
            libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
                "Failed to set signal handler for SIGTERM. Error: %d.", errno);
            return (EXIT_FAILURE);
        }

        if (signal(SIGINT, &globalShutdownSignalHandler) == SIG_ERR) {
            libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
                "Failed to set signal handler for SIGINT. Error: %d.", errno);
            return (EXIT_FAILURE);
        }
    #else
        struct sigaction shutdownAction;
        
        shutdownAction.sa_handler = &DVSDataConv::globalShutdownSignalHandler;
        shutdownAction.sa_flags   = 0;
        sigemptyset(&shutdownAction.sa_mask);
        sigaddset(&shutdownAction.sa_mask, SIGTERM);
        sigaddset(&shutdownAction.sa_mask, SIGINT);

        if (sigaction(SIGTERM, &shutdownAction, NULL) == -1) {
            libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
                "Failed to set signal handler for SIGTERM. Error: %d.", errno);
            raise (EXIT_FAILURE);
        }

        if (sigaction(SIGINT, &shutdownAction, NULL) == -1) {
            libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
                "Failed to set signal handler for SIGINT. Error: %d.", errno);
            raise (EXIT_FAILURE);
        }
    #endif

    //libcaer::devices::davis davisHandle = libcaer::devices::davis(1);

    //davisHandle = libcaer::devices::davis(ID);

    // Let's take a look at the information we have on the device.
    struct caer_davis_info davis_info = davisHandle.infoGet();

    printf("%s --- ID: %d, Master: %d, DVS X: %d, DVS Y: %d, Logic: %d.\n", davis_info.deviceString,
        davis_info.deviceID, davis_info.deviceIsMaster, davis_info.dvsSizeX, davis_info.dvsSizeY,
        davis_info.logicVersion);

    // Send the default configuration before using the device.
    // No configuration is sent automatically!
    davisHandle.sendDefaultConfig();
    

    // Tweak some biases, to increase bandwidth in this case.
    struct caer_bias_coarsefine coarseFineBias;

    coarseFineBias.coarseValue        = 2;
    coarseFineBias.fineValue          = 116;
    coarseFineBias.enabled            = true;
    coarseFineBias.sexN               = false;
    coarseFineBias.typeNormal         = true;
    coarseFineBias.currentLevelNormal = true;


    davisHandle.configSet(DAVIS_CONFIG_BIAS, DAVIS240_CONFIG_BIAS_PRBP, caerBiasCoarseFineGenerate(coarseFineBias));

    coarseFineBias.coarseValue        = 1;
    coarseFineBias.fineValue          = 33;
    coarseFineBias.enabled            = true;
    coarseFineBias.sexN               = false;
    coarseFineBias.typeNormal         = true;
    coarseFineBias.currentLevelNormal = true;


    davisHandle.configSet(DAVIS_CONFIG_BIAS, DAVIS240_CONFIG_BIAS_PRSFBP, caerBiasCoarseFineGenerate(coarseFineBias));

    // Set parsing intervall where container interval is in [10μs] unit
    davisHandle.configSet(CAER_HOST_CONFIG_PACKETS, CAER_HOST_CONFIG_PACKETS_MAX_CONTAINER_INTERVAL, container_interval);

    // Set number of events per packet
    //davisHandle.configSet(CAER_HOST_CONFIG_PACKETS, CAER_HOST_CONFIG_PACKETS_MAX_CONTAINER_PACKET_SIZE, container_interval);

    // Let's verify they really changed!
    uint32_t prBias   = davisHandle.configGet(DAVIS_CONFIG_BIAS, DAVIS240_CONFIG_BIAS_PRBP);
    uint32_t prsfBias = davisHandle.configGet(DAVIS_CONFIG_BIAS, DAVIS240_CONFIG_BIAS_PRSFBP);

    printf("New bias values --- PR-coarse: %d, PR-fine: %d, PRSF-coarse: %d, PRSF-fine: %d.\n",
    caerBiasCoarseFineParse(prBias).coarseValue, caerBiasCoarseFineParse(prBias).fineValue,
    caerBiasCoarseFineParse(prsfBias).coarseValue, caerBiasCoarseFineParse(prsfBias).fineValue);

}

// Now let's get start getting some data from the device. We just loop in blocking mode,
// no notification needed regarding new events. The shutdown notification, for example if
// the device is disconnected, should be listened to.
void DVSDataConv::startdatastream(){

    // Let's get configs about buffer
    davisHandle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BUFFER_SIZE, buffer_size);

    uint32_t BFSize   = davisHandle.configGet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BUFFER_SIZE);

    printf("Buffer size: %d", BFSize);

    davisHandle.dataStart(nullptr, nullptr, nullptr, &DVSDataConv::usbShutdownHandler, nullptr);

    // Let's turn on blocking data-get mode to avoid wasting resources.
    davisHandle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);
 
}


// Process an event and convert any given PolarityEvents to sparse tensor structure
torch::Tensor DVSDataConv::update(){
    std::vector<DVSevents::PolarityEvent> polarity_events;
    std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = nullptr;

    do {
      packetContainer = davisHandle.dataGet();
    } while (packetContainer == nullptr);


    printf("\nGot event container with %d packets (allocated).\n", packetContainer->size());

    for (auto &packet : *packetContainer) {
        if (packet == nullptr) {
            printf("Packet is empty (not present).\n");
            continue; // Skip if nothing there.
        }

        printf("Packet of type %d -> %d events, %d capacity.\n", packet->getEventType(), packet->getEventNumber(), packet->getEventCapacity());

        if (packet->getEventType() == POLARITY_EVENT) {
            std::shared_ptr<const libcaer::events::PolarityEventPacket> polarity = std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);
            // Print out timestamps and addresses.

            printf("Lowest Timestamp: %f\n", (float (packetContainer->getLowestEventTimestamp())/1000000));
            printf("Highest Timestamp: %f\n", (float (packetContainer->getHighestEventTimestamp())/1000000));
            printf("Time span of polarity packet: %ld\n", (packetContainer->getHighestEventTimestamp() - packetContainer->getLowestEventTimestamp()));

            for (const auto &evt : *polarity) {
                if (evt.isValid() == true){
                    DVSevents::PolarityEvent polarity_event;

                    polarity_event.timestamp = evt.getTimestamp64(*polarity);
                    polarity_event.x = evt.getX();
                    polarity_event.y = evt.getY();
                    polarity_event.polarity   = evt.getPolarity();

                    polarity_events.push_back(polarity_event);

                    //printf("Time: %d\n", polarity_event.timestamp);
                    //printf("x: %d\n", polarity_event.x);
                    //printf("y: %d\n", polarity_event.y);
                    //printf("polarity: %d\n", polarity_event.polarity);
                }
            }
        }
    }

    printf("Polarity event processed! Convert to tensor...\n");
    auto event_tensors = convert_polarity_events(polarity_events);
    std::cout << event_tensors.sizes() << std::endl;

    return event_tensors;
}

// Stops the datastream
int DVSDataConv::stopdatastream(){
    davisHandle.dataStop();
    // Close automatically done by destructor.
    printf("Shutdown successful.\n");
    return (EXIT_FAILURE);
}

/**
 * Converts a vector of polarity events into sparse tensors.
 * This function is taken from norse/AEDAT (https://github.com/norse/aedat).
 * 
 * Copyright (c) 2020 Christian Pehle and Jens. E. Pedersen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */
torch::Tensor DVSDataConv::convert_polarity_events(std::vector<DVSevents::PolarityEvent> &polarity_events, const std::vector<int64_t> &tensor_size) {
    const size_t size = polarity_events.size();
    std::vector<int64_t> indices(3 * size);
    std::vector<int8_t> values;
    const auto max_duration =
        tensor_size.empty()
            ? polarity_events.back().timestamp - polarity_events[0].timestamp
            : tensor_size[0];

    for (size_t idx = 0; idx < size; idx++) {
        auto event = polarity_events[idx];
        auto event_time = event.timestamp - polarity_events[0].timestamp;

        indices[idx] = event_time;
        indices[size + idx] = event.x;
        indices[2 * size + idx] = event.y;
        values.push_back(event.polarity ? 1 : -1);
    }

    auto index_options = torch::TensorOptions().dtype(torch::kInt64);
    torch::Tensor ind = torch::from_blob(indices.data(), {3, static_cast<uint32_t>(size)}, index_options);

    auto value_options = torch::TensorOptions().dtype(torch::kInt8);
    torch::Tensor val = torch::from_blob(values.data(), {static_cast<uint32_t>(size)}, value_options);

    auto events =
        tensor_size.empty()
            ? torch::sparse_coo_tensor(ind, val)
            : torch::sparse_coo_tensor(ind, val, torch::IntArrayRef(tensor_size));

    return events.clone();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<DVSDataConv>(m, "DVSDataConv")
      .def(py::init<int &, int &>())
      .def("connect2camera", &DVSDataConv::connect2camera)
      .def("startdatastream", &DVSDataConv::startdatastream)
      .def("update", &DVSDataConv::update)
      .def("stopdatastream", &DVSDataConv::stopdatastream);

}


