#!/usr/bin/env bash

# pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
# TORCH_PATH=$(python -c "import os,torch; print(os.path.dirname(torch.__file__) + \"/lib/\")")
INSTALL_PATH=$(find _skbuild -name "aestream" -type d | grep install)
AESTREAM_PATH=$(readlink -f $INSTALL_PATH)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${AESTREAM_PATH}"
