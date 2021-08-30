#pragma once

#include <fstream>
#include <iostream>
#include <map>
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
#include "generator.hpp"
#include "imus_generated.h"
#include "ioheader_generated.h"
#include "rapidxml.hpp"
#include "trigger_generated.h"

struct AEDAT4 {
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

  void load(const std::string &filename) {
    struct stat stat_info;

    auto fd = open(filename.c_str(), O_RDONLY, 0);

    if (fd < 0) {
      throw std::runtime_error("Failed to open file");
    }

    if (fstat(fd, &stat_info)) {
      throw std::runtime_error("Failed to stat file");
    }

    char *data = static_cast<char *>(
        mmap(NULL, stat_info.st_size, PROT_READ, MAP_SHARED, fd, 0));
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

    // std::cout << ioheader->infoNode()->str() << std::endl;

    doc.parse<0>((char *)(ioheader->infoNode()->str().c_str()));

    // extract necessary data from XML
    auto node = doc.first_node();
    for (rapidxml::xml_node<> *outinfo = node->first_node(); outinfo;
         outinfo = outinfo->next_sibling()) {

      auto attributes = collect_attributes(outinfo);
      if (attributes["name"] != "outInfo") {
        continue;
      }

      for (rapidxml::xml_node<> *child = outinfo->first_node(); child;
           child = child->next_sibling()) {
        OutInfo info;
        auto attributes = collect_attributes(child);
        if (!attributes.contains("name")) {
          continue;
        }

        info.name = std::stoi(attributes["name"]);

        for (rapidxml::xml_node<> *attr = child->first_node(); attr;
             attr = attr->next_sibling()) {
          auto attributes = collect_attributes(attr);
          if (attributes["key"] == "compression") {
            info.compression = attr->value();
          } else if (attributes["key"] == "typeIdentifier") {
            info.type = OutInfo::to_type(attr->value());
          } else if (attributes["name"] == "info") {
            for (rapidxml::xml_node<> *info_node = attr->first_node();
                 info_node; info_node = info_node->next_sibling()) {
              auto infos = collect_attributes(info_node);

              if (infos["key"] == "sizeX") {
                info.size_x = std::stoi(info_node->value());
              } else if (infos["key"] == "sizeY") {
                info.size_y = std::stoi(info_node->value());
              }
            }
          }
        }
        outinfos.push_back(info);
      }
    }

    // for (auto info : outinfos) {
    //   std::cout << "{" << info.name << ", " << info.compression << ", "
    //             << info.type << ", " << info.size_x << ", " << info.size_y
    //             << "}" << std::endl;
    // }

    size_t data_table_position = ioheader->dataTablePosition();

    data += ioheader_offset + 4;

    // we have to treat each packet according the compression method used,
    // which can be found in ioheader->compression()
    // assume LZ4 compression for now

    const size_t dst_size_fixed = 10000000;
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

    uint64_t count = 0;
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
        return;
      }

      switch (stream_id) {
      // switch (outinfos[stream_id].type) {
      // case OutInfo::Type::EVTS: {
      case 0: {
        auto event_packet = GetSizePrefixedEventPacket(&dst_buffer[0]);
        for (auto event : *event_packet->elements()) {
          count += 1;
          const auto e = AEDAT::PolarityEvent{
              static_cast<uint64_t>(event->t()),
              static_cast<uint16_t>(event->x()),
              static_cast<uint16_t>(event->y()),
              1,
              static_cast<bool>(event->on()),
          };
          if (e.x > 640 || e.y > 480) {
            printf("Wrong coords (%lu) %lu: %ux%u\n", count, e.timestamp, e.x,
                   e.y);
          }
          // printf("Event:%lu: %3d  %3d\n", e.timestamp, e.x, e.y);
          polarity_events.push_back(e);
        }
        break;
      }
      case OutInfo::Type::FRME: {
        Frame res;
        auto frame_packet = GetSizePrefixedFrame(&dst_buffer[0]);
        res.time = frame_packet->t();
        res.width = frame_packet->width();
        res.height = frame_packet->height();

        // std::cout << frame_packet->offset_y() << std::endl;

        auto pixels = frame_packet->pixels()->data();

        // res.pixels.reserve(res.width * res.height * 3);

        for (int j = 0; j < res.height; j++) {
          for (int i = 0; i < res.width; i++) {
            res.pixels.push_back(pixels[i + j * res.width]);
            res.pixels.push_back(pixels[i + j * res.width]);
            res.pixels.push_back(pixels[i + j * res.width]);
          }
        }

        frames.push_back(res);
        break;
      }
      case OutInfo::Type::IMUS: {
        auto imu_packet = GetSizePrefixedImuPacket(&dst_buffer[0]);
        break;
      }
      case OutInfo::Type::TRIG: {
        auto trigger_packet = GetSizePrefixedTriggerPacket(&dst_buffer[0]);
        break;
      }
      default: {
        break;
      }
      }
    }
  }

  AEDAT4() {}

  AEDAT4(const std::string &filename) { load(filename); }

  std::vector<OutInfo> outinfos;
  std::vector<Frame> frames;
  std::vector<AEDAT::PolarityEvent> polarity_events;
};