#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <lz4.h>
#include <lz4frame.h>

#include "aedat.hpp"
#include "events_generated.h"
#include "file_data_table_generated.h"
#include "frame_generated.h"
#include "imus_generated.h"
#include "ioheader_generated.h"
#include "trigger_generated.h"

struct AEDAT4 {

  void load(std::string filename) {
    struct stat stat_info;

    auto fd = open(filename.c_str(), O_RDONLY, 0);

    if (fd < 0) {
      std::cout << "Failed to open file" << std::endl;
      return;
    }

    if (fstat(fd, &stat_info)) {
      std::cout << "Failed to stat" << std::endl;
      return;
    }

    char *data = static_cast<char *>(
        mmap(NULL, stat_info.st_size, PROT_READ, MAP_SHARED, fd, 0));
    char *buffer_start = data;

    auto header = std::string(data, 14);
    data += 14; // size of the header

    // find size of IOHeader (it is variable)
    flatbuffers::uoffset_t ioheader_offset =
        *reinterpret_cast<flatbuffers::uoffset_t *>(data);
    const IOHeader *ioheader = GetSizePrefixedIOHeader(data);

    std::cout << ioheader->infoNode()->str() << std::endl;

    size_t data_table_position = ioheader->dataTablePosition();

    data += ioheader_offset + 4;
    char *header_end = data;

    // we have to treat each packet according the compression method used,
    // which can be found in ioheader->compression()
    // assume LZ4 compression for now

    const size_t dst_size_fixed = 1000000;
    std::vector<uint8_t> dst_buffer(dst_size_fixed);
    LZ4F_decompressionContext_t ctx;
    LZ4F_errorCode_t lz4_error =
        LZ4F_createDecompressionContext(&ctx, LZ4F_VERSION);

    if (LZ4F_isError(lz4_error)) {
      printf("Decompression error: %s\n", LZ4F_getErrorName(lz4_error));
      return;
    }

    size_t dst_size = dst_size_fixed;
    char *data_table_start = buffer_start + data_table_position;
    size_t data_table_size = stat_info.st_size - data_table_position;

    auto ret = LZ4F_decompress(ctx, &dst_buffer[0], &dst_size, data_table_start,
                               &data_table_size, nullptr);
    if (LZ4F_isError(ret)) {
      printf("Decompression error: %s\n", LZ4F_getErrorName(ret));
      return;
    }

    auto file_data_table = GetSizePrefixedFileDataTable(&dst_buffer[0]);

    for (auto elem : *file_data_table->Table()) {
      // just here to illustrate access to the data
    }

    while (data < buffer_start + data_table_position) {
      int32_t stream_id = *reinterpret_cast<int32_t *>(data);
      data += 4;
      size_t size = *reinterpret_cast<int32_t *>(data);
      data += 4;

      size_t dst_size = dst_size_fixed;
      auto ret =
          LZ4F_decompress(ctx, &dst_buffer[0], &dst_size, data, &size, nullptr);
      data += size;

      if (LZ4F_isError(ret)) {
        printf("Decompression error: %s\n", LZ4F_getErrorName(ret));
        continue;
      }

      if (stream_id == 0) {
        auto event_packet = GetSizePrefixedEventPacket(&dst_buffer[0]);
        for (auto event : *event_packet->elements()) {
          polarity_events.push_back(
              AEDAT::PolarityEvent{1, static_cast<uint32_t>(event->on()),
                                   static_cast<uint32_t>(event->x()),
                                   static_cast<uint32_t>(event->y()),
                                   static_cast<uint32_t>(event->t())});
        }
      }
      if (stream_id == 1) {
        auto frame_packet = GetSizePrefixedFrame(&dst_buffer[0]);
        // std::cout << frame_packet->t() << " "
        // 	  << frame_packet->width() << " "
        //	  << frame_packet->height() << " "
        //	  << std::endl;
      }
      if (stream_id == 2) {
        auto imu_packet = GetSizePrefixedImuPacket(&dst_buffer[0]);
      }
      if (stream_id == 3) {
        auto trigger_packet = GetSizePrefixedTriggerPacket(&dst_buffer[0]);
      }
    }
  }

  std::vector<AEDAT::PolarityEvent> polarity_events;
};

int main() {
  AEDAT4 dat;
  dat.load("example_data/kth/example.aedat4");
}
