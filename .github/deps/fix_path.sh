#!/usr/bin/env bash

INSTALL_PATH=$(find _skbuild -name 'aestream' -type d | grep install)
AESTREAM_PATH=$(readlink -f "$INSTALL_PATH")
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${AESTREAM_PATH}"
