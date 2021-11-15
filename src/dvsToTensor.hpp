#ifndef DVSTOTENSOR_HPP
#define DVSTOTENSOR_HPP

#include "generator.hpp"
#include "usb.hpp"
#include "convert.hpp"

Generator<torch::Tensor>
sparse_tensor_generator(std::string camera, std::uint16_t deviceId, std::uint8_t deviceAddress); 

Generator<torch::Tensor>
sparse_tensor_generator(const std::string serial_number = "None");

#endif