#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <lz4.h>
#include <lz4frame.h>

#include <flatbuffers/flatbuffers.h>

#include "utils.hpp"
#include <aer.hpp>

#include "aedat.hpp"
#include "events_generated.h"
#include "file_data_table_generated.h"
#include "frame_generated.h"
#include "generator.hpp"
#include "imus_generated.h"
#include "ioheader_generated.h"
#include "rapidxml.hpp"
#include "trigger_generated.h"

struct AEDAT4 : FileBase {

  struct Frame {
    int64_t time;
    int16_t width;
    int16_t height;
    std::vector<uint8_t> pixels;
  };

  struct OutInfo {
    enum Type { EVTS, FRME, IMUS, TRIG };
    int name;
    int size_x;
    int size_y;
    Type type;
    std::string compression;

    static Type to_type(std::string str) {
      if (str == "EVTS") {
        return Type::EVTS;
      } else if (str == "FRME") {
        return Type::FRME;
      } else if (str == "IMUS") {
        return Type::IMUS;
      } else if (str == "TRIG") {
        return Type::TRIG;
      } else {
        throw std::runtime_error("unexpected event type");
      }
    }
  };

  std::map<std::string, std::string>
  collect_attributes(rapidxml::xml_node<> *node) {
    std::map<std::string, std::string> attributes;
    for (const rapidxml::xml_attribute<> *a = node->first_attribute(); a;
         a = a->next_attribute()) {
      auto name = std::string(a->name(), a->name_size());
      auto value = std::string(a->value(), a->value_size());
      attributes[name] = value;
    }
    return attributes;
  }

  void open_aedat() {
    struct stat stat_info;

    if (fstat(fileno(fp.get()), &stat_info)) {
      throw std::runtime_error("Failed to stat file");
    }

    char *data = static_cast<char *>(mmap(NULL, stat_info.st_size, PROT_READ,
                                          MAP_SHARED, fileno(fp.get()), 0));
    char *buffer_start = data;

    auto header = std::string(data, 14);
    if (header != "#!AER-DAT4.0\r\n") {
      throw std::runtime_error("Invalid AEDAT version");
    }

    data += 14; // size of the version string

    // find size of IOHeader (it is variable)
    flatbuffers::uoffset_t ioheader_offset =
        *reinterpret_cast<flatbuffers::uoffset_t *>(data);
    const IOHeader *ioheader = GetSizePrefixedIOHeader(data);

    rapidxml::xml_document<> doc;

    // doc.parse<0>((char *)(ioheader->info_node()->str().c_str()));

    // extract necessary data from XML
    // auto node = doc.first_node();
    // for (rapidxml::xml_node<> *outinfo = node->first_node(); outinfo;
    //      outinfo = outinfo->next_sibling()) {

    //   auto attributes = collect_attributes(outinfo);
    //   if (attributes["name"] != "outInfo") {
    //     continue;
    //   }

    //   for (rapidxml::xml_node<> *child = outinfo->first_node(); child;
    //        child = child->next_sibling()) {
    //     OutInfo info;
    //     auto attributes = collect_attributes(child);
    //     if (!attributes.contains("name")) {
    //       continue;
    //     }

    //     info.name = std::stoi(attributes["name"]);

    //     for (rapidxml::xml_node<> *attr = child->first_node(); attr;
    //          attr = attr->next_sibling()) {
    //       auto attributes = collect_attributes(attr);
    //       if (attributes["key"] == "compression") {
    //         info.compression = attr->value();
    //       } else if (attributes["key"] == "typeIdentifier") {
    //         info.type = OutInfo::to_type(attr->value());
    //       } else if (attributes["name"] == "info") {
    //         for (rapidxml::xml_node<> *info_node = attr->first_node();
    //              info_node; info_node = info_node->next_sibling()) {
    //           auto infos = collect_attributes(info_node);

    //           if (infos["key"] == "sizeX") {
    //             info.size_x = std::stoi(info_node->value());
    //           } else if (infos["key"] == "sizeY") {
    //             info.size_y = std::stoi(info_node->value());
    //           }
    //         }
    //       }
    //     }
    //     outinfos.push_back(info);
    //   }
    // }

    data_table_position = ioheader->data_table_position();

    if (data_table_position < 0) {
      throw std::runtime_error(
          "AEDAT files without datatables are currently not supported");
    }

    // Seek file pointer
    fseek(fp.get(), 14 + data_table_position + 4, SEEK_SET);

    // TODO: Re-introduce header file parsing
    // data += ioheader_offset + 4;

    // we have to treat each packet according the compression method used,
    // which can be found in ioheader->compression()
    // assume LZ4 compression for now

    // std::vector<uint8_t> dst_buffer(BUFFER_SIZE);
    // LZ4F_errorCode_t lz4_error =
    //     LZ4F_createDecompressionContext(&ctx, LZ4F_VERSION);

    // if (LZ4F_isError(lz4_error)) {
    //   printf("Decompression context error: %s\n",
    //   LZ4F_getErrorName(lz4_error)); return;
    // }

    // size_t dst_size = BUFFER_SIZE;
    // char *data_table_start = buffer_start + data_table_position;
    // size_t data_table_size = stat_info.st_size - data_table_position;

    // auto ret = LZ4F_decompress(ctx, &dst_buffer[0], &dst_size,
    // data_table_start,
    //                            &data_table_size, nullptr);
    // if (LZ4F_isError(ret)) {
    //   printf("Decompression DataTable error: %s\n", LZ4F_getErrorName(ret));
    //   return;
    // }

    // auto file_data_table = GetSizePrefixedFileDataTable(&dst_buffer[0]);

    // for (auto elem : *file_data_table->table()) {
    //   // just here to illustrate access to the data
    // }
  }

  std::tuple<AER::Event *, size_t> read_events(const size_t &n_events) {
    std::vector<uint8_t> dst_buffer(BUFFER_SIZE);
    AER::Event *events = (AER::Event *)malloc(n_events * sizeof(AER::Event));
    uint64_t count = 0;
    do {
      int32_t stream_id;
      fread(&stream_id, sizeof(int32_t), 1, fp.get());
      size_t size;
      fread(&size, sizeof(int32_t), 1, fp.get());

      size_t dst_size = BUFFER_SIZE;
      auto ret = LZ4F_decompress(ctx, &dst_buffer[0], &dst_size, fp.get(),
                                 &size, nullptr);
      fseek(fp.get(), size, SEEK_SET);

      if (LZ4F_isError(ret)) {
        printf("Decompression package error: %s\n", LZ4F_getErrorName(ret));
        return {nullptr, 0};
      }

      switch (stream_id) {
      // switch (outinfos[stream_id].type) {
      // case OutInfo::Type::EVTS: {
      case 0: {
        auto event_packet = GetSizePrefixedEventPacket(&dst_buffer[0]);
        for (auto event : *event_packet->elements()) {
          const auto e = AER::Event{
              static_cast<uint64_t>(event->t()),
              static_cast<uint16_t>(event->x()),
              static_cast<uint16_t>(event->y()),
              static_cast<bool>(event->on()),
          };
          events[count] = e;
          count += 1;
        }
        break;
      }
      case OutInfo::Type::FRME: {
        // Frame res;
        // auto frame_packet = GetSizePrefixedFrame(&dst_buffer[0]);
        // res.time = frame_packet->t();
        // res.width = frame_packet->width();
        // res.height = frame_packet->height();

        // auto pixels = frame_packet->pixels()->data();

        // for (int j = 0; j < res.height; j++) {
        //   for (int i = 0; i < res.width; i++) {
        //     res.pixels.push_back(pixels[i + j * res.width]);
        //     res.pixels.push_back(pixels[i + j * res.width]);
        //     res.pixels.push_back(pixels[i + j * res.width]);
        //   }
        // }

        // frames.push_back(res);
        break;
      }
      case OutInfo::Type::IMUS: {
        // auto imu_packet = GetSizePrefixedImuPacket(&dst_buffer[0]);
        break;
      }
      case OutInfo::Type::TRIG: {
        // auto trigger_packet = GetSizePrefixedTriggerPacket(&dst_buffer[0]);
        break;
      }
      default: {
        break;
      }
      }
    } while (ftell(fp.get()) < data_table_position && n_events - count > 0);
    return {events, count};
  }

  static std::tuple<char *, size_t> compress_lz4(char *buffer, size_t size) {
    auto lz4FrameBound = LZ4F_compressBound(size, nullptr);
    auto new_buffer = new char[lz4FrameBound];

    auto compression =
        LZ4F_compressFrame(new_buffer, lz4FrameBound, buffer, size, nullptr);
    return {new_buffer, compression};
    // LZ4F_compressionContext_t ctx;
    // auto contextCreation = LZ4F_createCompressionContext(&ctx,
    // LZ4F_VERSION);
    // auto headerSize =
    //     LZ4F_compressBegin(ctx, new_buffer, lz4FrameBound, nullptr);
    // if (LZ4F_isError(headerSize)) {
    //   std::stringstream err;
    //   err << "Compression error: " << LZ4F_getErrorName(headerSize);
    //   throw std::runtime_error(err.str());
    // }

    // auto packetSize = LZ4F_compressUpdate(ctx, new_buffer, lz4FrameBound,
    //                                       buffer, size, nullptr);
    // auto footerSize = LZ4F_compressEnd(ctx, new_buffer, lz4FrameBound,
    // nullptr);
    // return {new_buffer, headerSize + packetSize + footerSize};
  }

  static size_t save_header(std::fstream &stream) {
    stream << "#!AER-DAT4.0\r\n";
    // Save header
    flatbuffers::FlatBufferBuilder fbb;
    fbb.ForceDefaults(true);
    auto infoNode = "<dv version=\"4.0\"><node name=\"outInfo\"></node></dv>";
    auto headerOffset =
        CreateIOHeaderDirect(fbb, CompressionType_LZ4, -1L, infoNode);
    fbb.FinishSizePrefixed(headerOffset);
    stream.write((char *)fbb.GetBufferPointer(), fbb.GetSize());
    std::cout << "Data " << stream.tellp() << std::endl;
    return fbb.GetSize();
  }

  static void save_footer(std::fstream &stream, size_t headerSize,
                          int64_t timestampStart, int64_t timestampEnd,
                          size_t eventCount) {
    flatbuffers::FlatBufferBuilder fbb;
    // Mutate offset to data table
    size_t tableOffset = stream.tellp();
    stream.seekg(14); // 14 bytes for version
    auto length = headerSize;
    char *data = new char[length];
    stream.read(data, length);
    auto header = GetSizePrefixedIOHeader(data);
    auto new_header = CreateIOHeaderDirect(
        fbb, header->compression(), tableOffset, header->info_node()->c_str());
    fbb.FinishSizePrefixed(new_header);
    stream.seekp(14); // 14 bytes for version
    stream.write((char *)fbb.GetBufferPointer(), fbb.GetSize());
    stream.seekp(tableOffset);

    // Write data table
    fbb.Clear();
    auto defBuilder = FileDataDefinitionBuilder(fbb);
    auto packetHeader = PacketHeader(0, eventCount);
    defBuilder.add_packet_info(&packetHeader);
    defBuilder.add_num_elements(1);
    defBuilder.add_timestamp_start(timestampStart);
    defBuilder.add_timestamp_end(timestampEnd);
    auto dataDefinition = defBuilder.Finish();
    auto tables = std::vector{dataDefinition};
    auto tableVector = fbb.CreateVector(tables);
    auto dataTable = CreateFileDataTable(fbb, tableVector);
    fbb.FinishSizePrefixed(dataTable);
    auto [compressed, size] =
        compress_lz4((char *)fbb.GetBufferPointer(), fbb.GetSize());
    stream.write(compressed, size);
    std::cout << "Table " << tableOffset << std::endl;
  }

  static void save_events(std::fstream &stream,
                          std::vector<AEDAT::PolarityEvent> events) {
    // Create event buffer
    flatbuffers::FlatBufferBuilder fbb;
    fbb.ForceDefaults(true);
    std::vector<Event> bufferEvents;
    for (auto event : events) {
      const Event e = Event(static_cast<int64_t>(event.timestamp),
                            static_cast<int16_t>(event.x),
                            static_cast<int16_t>(event.y), event.polarity);
      bufferEvents.push_back(e);
    }
    auto eventVector = CreateEventPacketDirect(fbb, &bufferEvents);
    FinishSizePrefixedEventPacketBuffer(fbb, eventVector);
    auto [compressed, size] =
        compress_lz4((char *)fbb.GetBufferPointer(), fbb.GetSize());

    // Write packet header
    auto packetHeader = PacketHeader(0, size);
    stream.write((char *)&packetHeader, 8);

    // Write events
    stream.write(compressed, size);
  }

  Generator<AER::Event> stream(size_t n_events = -1) {
    size_t size = 0;
    size_t count = 0;
    static const size_t STREAM_BUFFER_SIZE = 128;
    do {
      auto [events, size] = read_events(STREAM_BUFFER_SIZE);
      for (size_t i; i < size; i++) {
        co_yield events[i];
        count++;
      }
    } while (size > 0 && count - n_events > 0);
  }

  AEDAT4(const file_t &fp) : fp(fp) { open_aedat(); }
  AEDAT4(const std::string &filename) : fp(open_file(filename)) {
    open_aedat();
  }

private:
  const file_t &fp;
  uint64_t data_table_position;
  LZ4F_decompressionContext_t ctx;
};
