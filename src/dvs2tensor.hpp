#pragma once

#define LIBCAER_FRAMECPP_OPENCV_INSTALLED 0

#include <libcaercpp/devices/davis.hpp>
#include "DVSEvents.hpp"
#include <torch/script.h>
#include <atomic>
#include <csignal>

#ifndef DVSData_H
#define DVSData_H

using namespace std;


class DVSDataConv{
    public:
        // Parameters
        uint32_t container_interval;
        uint32_t buffer_size;
        libcaer::devices::davis davisHandle = libcaer::devices::davis(1); 

        DVSDataConv(uint32_t interval, uint32_t bfsize);

        static void globalShutdownSignalHandler(int signal);
        static void usbShutdownHandler(void *ptr); 
        void connect2camera(int ID);
        void startdatastream();
        torch::Tensor update();
        int stopdatastream();
        torch::Tensor convert_polarity_events(std::vector<DVSevents::PolarityEvent> &polarity_events, const std::vector<int64_t> &tensor_size = {});
};

#endif