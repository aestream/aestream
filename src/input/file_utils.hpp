#pragma once

#include <memory>
#include <stdio.h>

void close_file(FILE *fp);

typedef std::unique_ptr<FILE, decltype(&close_file)> unique_file_t;

unique_file_t open_file(const std::string &filename);

bool ends_with(std::string const &value, std::string const &ending);

#define HEADER_START 0x25
#define HEADER_END 0x0A
#define BUFFER_SIZE 4096