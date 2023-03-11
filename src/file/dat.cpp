#include "dat.hpp"

size_t dat_read_header(shared_file_t &fp) {
  size_t bytes_read = 0;
  uint8_t c, header_begins;

  do {
    header_begins = 1;
    do {
      bytes_read += fread(&c, 1, 1, fp.get());
      if (header_begins && c != HEADER_START) {
        if (fseek(fp.get(), -1, SEEK_CUR) != 0) {
          throw std::runtime_error("Failed to process .dat file header");
        }
        --bytes_read;
        goto process_length;
      } else {
        header_begins = 0;
      }
    } while (c != HEADER_END);
  } while (1);

  if (bytes_read == 0) {
    throw std::runtime_error("Failed to process .dat file header");
  }

process_length:
  fseek(fp.get(), 0, SEEK_END);
  size_t fileLength = ftell(fp.get());
  fseek(fp.get(), bytes_read + 2, SEEK_SET); // Skip header bytes
  return (fileLength - bytes_read) / sizeof(int64_t);
}

std::tuple<AER::Event *, size_t> dat_read_n_events(shared_file_t &fp,
                                                   const size_t &n_events) {
  const size_t buffer_size = BUFFER_SIZE > n_events ? n_events : BUFFER_SIZE;

  uint64_t *buffer = (uint64_t *)malloc(buffer_size * sizeof(uint64_t));
  AER::Event *events = (AER::Event *)malloc(n_events * sizeof(AER::Event));
  uint64_t timestep = 0;
  size_t overflows = 0;
  size_t index = 0;
  size_t size = 0;
  do {
    size = fread(buffer, sizeof(*buffer), buffer_size, fp.get());

    if (size == 0 && !feof(fp.get())) {
      throw std::runtime_error("Error when processing .dat file");
    }

    if (size > n_events - index) {
      size = n_events - index;
    }

    for (int i = 0; i < size; i++) {
      AER::Event event = dat_decode_event(buffer[i], overflows);
      if (event.timestamp < timestep) { // Timestep overflow occurred
        overflows++;
        event.timestamp = (overflows << 32) | event.timestamp;
      }
      events[index + i] = event;
    }
    index += size;
  } while (size > 0 && n_events - index > 0);
  return {events, index};
}

Generator<AER::Event> dat_stream_events(const std::string filename) {
  shared_file_t fp = open_file(filename);
  const auto n_events = dat_read_header(fp);
  return dat_stream_events(fp, n_events);
}

Generator<AER::Event> dat_stream_events(shared_file_t fp,
                                        const size_t n_events) {
  uint64_t buffer[BUFFER_SIZE];
  size_t count = 0;
  uint64_t timestep = 0;
  size_t overflows = 0;
  int n = 0;
  size_t size;
  do {
    size = fread(buffer, sizeof(uint64_t), BUFFER_SIZE, fp.get());

    if (size == 0 && !feof(fp.get())) {
      throw std::runtime_error("Error when processing .dat file");
    }

    for (int i = 0; i < size; i++) {
      AER::Event event = dat_decode_event(buffer[i], overflows);
      if (event.timestamp < timestep) { // Timestep overflow occurred
        overflows++;
        event.timestamp = (overflows << 32) | event.timestamp;
      }
      co_yield event;
    }
  } while (size > 0);
}

inline AER::Event dat_decode_event(uint64_t data, size_t overflows) {
  auto event = *(DATEvent *)(&data);
  return AER::Event{static_cast<uint64_t>(event.ts),
                    static_cast<uint16_t>(event.x),
                    static_cast<uint16_t>(event.y), static_cast<bool>(event.p)};

  static const uint64_t mask_4b = 0xFU, mask_14b = 0x3FFFU,
                        mask_32b = 0xFFFFFFFFU;
  const uint64_t lower = data & mask_32b;
  const uint64_t upper = data >> 32;
  const uint64_t timestep = (overflows << 32) | lower;

  return AER::Event{timestep, static_cast<uint16_t>(upper & mask_14b),
                    static_cast<uint16_t>((upper >> 14) & mask_14b),
                    ((upper >> 28) & mask_4b) > 0};
}
