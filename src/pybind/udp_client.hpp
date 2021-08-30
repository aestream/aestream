#pragma once

#include <netdb.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

int udp_client(std::string port);