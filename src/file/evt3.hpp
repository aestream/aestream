#pragma once

#include <optional>

#include "../aer.hpp"
#include "../generator.hpp"

#include "utils.hpp"

struct EVT3 : FileBase {

  struct RawEvent {
    unsigned int content : 12;
    unsigned int type : 4;
  } __attribute__((packed));

  struct EventCoordinate {
    unsigned int coordinate : 11;
    bool meta : 1; // Either system type (EVT_ADDR_Y) or polarity (EVT_ADDR_X)
    unsigned int type : 4;
  } __attribute__((packed));

  struct Vect8 {
    unsigned int valid : 8;
    unsigned int unused : 4;
    unsigned int type : 4;
  } __attribute__((packed));

  struct Vect12 {
    unsigned int valid : 12;
    unsigned int type : 4;
  } __attribute__((packed));

  struct Continued4 {
    unsigned int content : 4;
    unsigned int unused : 8;
    unsigned int type : 4;
  } __attribute__((packed));

  struct Continued12 {
    unsigned int content : 12;
    unsigned int type : 4;
  } __attribute__((packed));

  struct ContainerProcessingState {
    // Current bits to read from
    uint16_t bits;
    // Total number of bits in event
    uint16_t bit_size;
    // Bits to still process
    size_t bits_remaining;
  };

  // https://docs.prophesee.ai/stable/data/encoding_formats/evt3.html
  enum EventType {
    EVT_ADDR_Y = 0b0000,
    EVT_ADDR_X = 0b0010,
    VEC_BASE_X = 0b0011,
    VECT_12 = 0b0100,
    VECT_8 = 0b0101,
    EVT_TIME_LOW = 0b0110,
    CONTINUED_4 = 0b0111,
    EVT_TIME_HIGH = 0b1000,
    EXT_TRIDDER = 0b1010,
    OTHERS = 0b1110,
    CONTINUED_12 = 0b1111
  };

  std::tuple<std::vector<AER::Event>, size_t>
  read_events(const int64_t n_events = -1) {
    static const size_t READ_BUFFER_SIZE = 4096;
    const size_t buffer_size = n_events > 0 && n_events < READ_BUFFER_SIZE
                                   ? n_events * 2
                                   : READ_BUFFER_SIZE;
    // If reading the full file we reserve bytes / 5 as a lower estimate
    const size_t event_array_size = n_events > 0 ? n_events : file_bytes / 5;

    std::vector<uint16_t> buffer_vector = std::vector<uint16_t>();
    buffer_vector.reserve(buffer_size);
    uint16_t *buffer = buffer_vector.data();

    std::vector<AER::Event> events{};
    events.reserve(event_array_size);

    size_t size = 0;
    do {
      size = fread(buffer, sizeof(uint16_t), buffer_size, fp.get());
      if (size == 0 && !feof(fp.get())) {
        throw std::runtime_error("Error when processing .evt3 file");
      }

      auto offset = decode_event_buffer(
          buffer, size, events,
          n_events < 0 ? buffer_size : n_events - events.size());
      fseek(fp.get(), offset * sizeof(uint16_t),
            SEEK_CUR); // Re-align file if we didn't process all events

    } while (size != 0 && (n_events < 0 || events.size() - n_events > 0));

    return {events, events.size()};
  }

  Generator<AER::Event> stream(const int64_t n_events = -1) {
    static const size_t STREAM_BUFFER_SIZE = 4096;
    int64_t size = 0, count = 0;
    std::vector<AER::Event> events;
    do {
      std::tie(events, size) = read_events(STREAM_BUFFER_SIZE);
      for (size_t i = 0; i < size; i++) {
        co_yield events[i];
        count++;
      }
    } while (size == STREAM_BUFFER_SIZE &&
             (n_events < 0 || n_events - count > 0));
  }

  explicit EVT3(const std::string &filename) : EVT3(open_file(filename)) {}
  explicit EVT3(file_t &&fp)
      : fp(std::move(fp)), file_bytes(file_size(this->fp.get())) {
    skip_evt3_header();
  }

private:
  static constexpr char HEADER_LINE_END = 0x0A;
  static constexpr char HEADER_LINE_START = 0x25;
  const file_t fp;
  const long file_bytes;
  bool is_first = true;
  uint32_t time_low = 0, time_high = 0, time_overflow_high = 0; // State
  uint64_t current_time = 0;
  std::optional<EventCoordinate> x_event = {};
  std::optional<EventCoordinate> y_event = {};
  ContainerProcessingState state = {};

  inline ContainerProcessingState
  process_vector_event(std::vector<AER::Event> &events,
                       int64_t remaining_events) {
    if (!y_event.has_value() || !x_event.has_value()) {
      throw std::runtime_error("EVT3 file format error: no header data for "
                               "base (x, y) coordinates given");
    }

    uint16_t y = y_event.value().coordinate;
    uint16_t i = state.bits_remaining;
    for (; i < state.bit_size; ++i) {
      if (state.bits & (1U << i)) {
        const AER::Event event = {
            current_time, static_cast<uint16_t>(x_event.value().coordinate), y,
            x_event.value().meta};
        events.push_back(std::move(event));
        remaining_events--;
      }
      x_event.value()
          .coordinate++; // Increment coordinate irregardless of validity
      if (remaining_events <= 0) {
        break;
      }
    }
    const uint16_t unprocessed = state.bit_size - i;
    return {
        state.bits, state.bit_size,
        unprocessed // Register number of unprocessed bits
    };
  }

  int32_t decode_event_buffer(uint16_t *buffer, size_t buffer_size,
                              std::vector<AER::Event> &events,
                              const size_t remaining_events) {
    const size_t max_remaining_limit =
        events.size() + remaining_events; // Maximum events to emit
    AER::Event event;
    EventCoordinate *x;
    Vect8 *vect8_event;
    Vect12 *vect12_event;

    // Process left-over vector event from previous call, if any
    if (state.bits_remaining > 0) {
      state = process_vector_event(events, remaining_events);
    }

    int64_t i = 0; // Buffer index
    for (; i < buffer_size; ++i) {
      auto raw_event = reinterpret_cast<RawEvent *>(&buffer[i]);
      switch (raw_event->type) {
      case EventType::EVT_ADDR_Y:
        y_event = *reinterpret_cast<EventCoordinate *>(raw_event);
        break;
      case EventType::EVT_ADDR_X:
        x = reinterpret_cast<EventCoordinate *>(raw_event);
        event = {current_time, static_cast<uint16_t>(x->coordinate),
                 static_cast<uint16_t>(y_event.value().coordinate), x->meta};
        events.push_back(event);
        break;
      case EventType::EVT_TIME_HIGH:
        static constexpr uint64_t TIMESTAMP_MAX = 1ULL << 11;
        time_high = raw_event->content;
        if (!is_first && time_high > TIMESTAMP_MAX + current_time) {
          time_overflow_high++;
        }
        if (is_first) { // Remove first flag
          is_first = false;
        }
        if (current_time & time_high != time_high) {
          time_low = 0;
        }
        current_time =
            (time_overflow_high << 24) | (time_high << 12) | time_low;
        break;
      case EventType::EVT_TIME_LOW:
        if (time_low < raw_event->content) {
          time_high++;
        }
        time_low += raw_event->content;
        current_time =
            (time_overflow_high << 24) | (time_high << 12) | time_low;
        break;
      case EventType::VEC_BASE_X:
        x_event = *reinterpret_cast<EventCoordinate *>(raw_event);
        break;
      case EventType::VECT_8:
        vect8_event = reinterpret_cast<Vect8 *>(raw_event);
        state.bits = vect8_event->valid;
        state.bit_size = 8;
        state =
            process_vector_event(events, max_remaining_limit - events.size());
        break;
      case EventType::VECT_12:
        vect12_event = reinterpret_cast<Vect12 *>(raw_event);
        state.bits = vect12_event->valid;
        state.bit_size = 12;
        state =
            process_vector_event(events, max_remaining_limit - events.size());
        break;
      }

      if (events.size() >=
          max_remaining_limit) { // Break if decoded enough events
        i++;
        break;
      }
    }
    return i - static_cast<int64_t>(
                   buffer_size); // Return offset for unprocessed events
  }

  void skip_evt3_header() {
    uint8_t c;
    while (1) {
      auto size = fread(&c, 1, 1, fp.get());
      if (size <= 0 || c != HEADER_LINE_START) {
        fseek(fp.get(), -1, SEEK_CUR); // Undo the last read
        break;
      }
      do {
        fread(&c, 1, 1, fp.get());
      } while (c != HEADER_LINE_END);
    }
  }
};