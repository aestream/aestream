#pragma once

#include <memory>

struct AER {
  struct Event {
    uint64_t timestamp;
    uint16_t x;
    uint16_t y;
    bool polarity;
  } __attribute__((packed));
};
