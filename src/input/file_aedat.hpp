#pragma once

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

struct AEDAT {
  enum class EventType : uint16_t {
    SPECIAL_EVENT = 0,
    POLARITY_EVENT = 1,
    FRAME_EVENT = 2,
    IMU6_EVENT = 3,
    IMU9_EVENT = 4,
    SPIKE_EVENT = 12,
  };

  struct PolarityEvent {
    uint64_t timestamp : 64;
    uint16_t x : 15;
    uint16_t y : 15;
    bool valid : 1;
    bool polarity : 1;
  } __attribute__((packed));

  struct SpecialEvent {
    uint32_t data : 24;
    uint8_t type : 7;
    bool valid : 1;
  } __attribute__((packed));

  struct DynapSEEvent {
    uint64_t timestamp : 64;
    uint32_t neuron_id : 20;
    uint16_t chip_id : 6;
    uint8_t core_id : 5;
    bool valid : 1;
  } __attribute__((packed));

  enum class SpecialEventType : uint8_t {
    TIMESTAMP_WRAP = 0,
    TIMESTAMP_RESET = 1,
    EXTERNAL_INPUT_RISING_EDGE = 2,
    EXTERNAL_INPUT_FALLING_EDGE = 3,
    EXTERNAL_INPUT_PULSE = 4,
    DVS_ROW_ONLY = 5,
    EXTERNAL_INPUT1_RISING_EDGE = 6,
    EXTERNAL_INPUT1_FALLING_EDGE = 7,
    EXTERNAL_INPUT1_PULSE = 8,
    EXTERNAL_INPUT2_RISING_EDGE = 9,
    EXTERNAL_INPUT2_FALLING_EDGE = 10,
    EXTERNAL_INPUT2_PULSE = 11,
    EXTERNAL_GENERATOR_RISING_EDGE = 12,
    EXTERNAL_GENERATOR_FALLING_EDGE = 13,
    APS_FRAME_START = 14,
    APS_FRAME_END = 15,
    APS_EXPOSURE_START = 16,
    APS_EXPOSURE_END = 17,
  };

  struct IMU6Event {
    uint64_t timestamp : 64;
    float accel_x;
    float accel_y;
    float accel_z;
    float gyro_x;
    float gyro_y;
    float gyro_z;
    float temp;
    uint32_t padding : 31;
    uint32_t valid : 1;
  } __attribute__((packed));

  struct IMU9Event {
    uint64_t timestamp : 64;
    float accel_x;
    float accel_y;
    float accel_z;
    float gyro_x;
    float gyro_y;
    float gyro_z;
    float temp;
    float comp_x;
    float comp_y;
    float comp_z;
    uint32_t padding : 31;
    uint32_t valid : 1;
  } __attribute__((packed));

  struct FrameEventHeader {
    uint32_t frame_start;
    uint32_t frame_end;
    uint32_t exposure_start;
    uint32_t exposure_end;
    uint32_t x_length;
    uint32_t y_length;
    uint32_t x_position;
    uint32_t y_position;
    uint32_t reserved : 17;
    uint32_t roi : 7;
    uint32_t filter : 4;
    uint32_t channels : 3;
    uint32_t valid : 1;
  } __attribute__((packed));

  struct FrameEvent {
    FrameEventHeader header;
    std::vector<uint16_t> pixels;
  };

  struct Header {
    EventType eventType;
    uint32_t eventSize;
    uint32_t eventTSOffset;
    uint32_t eventTSOverflow;
    uint32_t eventCapacity;
    uint32_t eventNumber;
    uint32_t eventValid;
    uint16_t eventSource;
  } __attribute__((packed));

  void load(const std::string &filename) {
    std::fstream fs;
    char line[128];
    Header header;
    std::string str = std::string(line);

    fs.open(filename, std::fstream::in);

    do {
      fs.getline(line, 128);
      str = std::string(line);
    } while (str.rfind("#!END-HEADER", 0) != 0);

    while (fs.read((char *)(&header), 28)) {
      if (header.eventTSOverflow != 0) {
        std::cout << "Unhandled TSOverflow "
                  << static_cast<uint16_t>(header.eventTSOverflow) << std::endl;
      }
      if (header.eventType == EventType::POLARITY_EVENT) {
        PolarityEvent polarity_event;
        for (size_t i = 0; i < header.eventNumber; i++) {
          fs.read((char *)(&polarity_event), header.eventSize);
          polarity_events.push_back(polarity_event);
        }
        fs.ignore((header.eventCapacity - header.eventNumber) *
                  header.eventSize);
      } else if (header.eventType == EventType::IMU6_EVENT) {
        IMU6Event imu6_event;
        for (size_t i = 0; i < header.eventNumber; i++) {
          fs.read((char *)(&imu6_event), header.eventSize);
          imu6_events.push_back(imu6_event);
        }
        fs.ignore((header.eventCapacity - header.eventNumber) *
                  header.eventSize);
      } else if (header.eventType == EventType::IMU9_EVENT) {
        IMU9Event imu9_event;
        for (size_t i = 0; i < header.eventNumber; i++) {
          fs.read((char *)(&imu9_event), header.eventSize);
          imu9_events.push_back(imu9_event);
        }
        fs.ignore((header.eventCapacity - header.eventNumber) *
                  header.eventSize);
      } else if (header.eventType == EventType::SPIKE_EVENT) {
        DynapSEEvent dyn_event;
        for (size_t i = 0; i < header.eventNumber; i++) {
          fs.read((char *)(&dyn_event), header.eventSize);
          dynapse_events.push_back(dyn_event);
        }
        fs.ignore((header.eventCapacity - header.eventNumber) *
                  header.eventSize);
      } else {
        std::cout << "Unhandled Event type "
                  << static_cast<uint16_t>(header.eventType) << std::endl;
        fs.ignore(header.eventCapacity * header.eventSize);
      }
    }
    return;
  }

  AEDAT() {}
  AEDAT(const std::string &filename) { load(filename); }

  std::vector<DynapSEEvent> dynapse_events;
  std::vector<IMU6Event> imu6_events;
  std::vector<IMU9Event> imu9_events;
  std::vector<PolarityEvent> polarity_events;
};