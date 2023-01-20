#!/usr/bin/env bash

# System dependencies
yum install -y lz4-devel ninja-build libgusb-devel gcc-10 || ( apt update && apt install -y cmake ninja-build g++-10 liblz4-dev libusb-1.0-0-dev )

# Install libcaer
curl -L -s -m 100 https://gitlab.com/inivation/dv/libcaer/-/archive/3.3.14/libcaer-3.3.14.tar.gz | tar zxf -
cd libcaer-3.3.14
cmake -DCMAKE_INSTALL_PREFIX=/usr . # Use specific cmake to avoid confusing versions
make -j 4 && make install
cd ..
