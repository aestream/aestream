#!/usr/bin/env bash

AESTREAM_PATH=$(find /opt -name 'aestream' -type d | grep lib)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${AESTREAM_PATH}"
