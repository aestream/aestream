#!/usr/bin/env bash

INSTALL_PATH=$(find /opt -name 'aestream' -type d)
AESTREAM_PATH=$(readlink -f "$INSTALL_PATH")
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${AESTREAM_PATH}"
