#pragma once
#include "aedat.hpp"
#include "aedat4.hpp"

#include <iostream>
#include <torch/script.h>

torch::Tensor
convert_polarity_events(std::vector<AEDAT::PolarityEvent> &polarity_events);
