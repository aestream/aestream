#pragma once

#include <atomic>

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/driver/camera.h>

#include "../aer.hpp"
#include "../generator.hpp"

Generator<AER::Event>
prophesee_event_generator(const std::atomic<bool> &runFlag,
                          const std::optional<std::string> serial_number);