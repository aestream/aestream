#pragma once

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/driver/camera.h>

#include "aedat.hpp"
#include "generator.hpp"

Generator<AEDAT::PolarityEvent>
prophesee_event_generator(const std::string serial_number);