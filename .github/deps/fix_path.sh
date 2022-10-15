#!/usr/bin/env bash

pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
TORCH_PATH=$(python3 -c "import os,torch; print(os.path.dirname(torch.__file__) + \"/lib/\")")
AESTREAM_PATH=$(find . -name "libaestream_lib.a" | head -n 1)
export LD_LIBRARY_PATH=${TORCH_PATH}:${AESTREAM_PATH}