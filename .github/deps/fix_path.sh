#!/usr/bin/env bash

pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
TORCH_PATH=$(python -c "import os,torch; print(os.path.dirname(torch.__file__) + \"/lib/\")")
AESTREAM_PATH=$(readlink -f $(dirname $(find _skbuild -name "libaedat4.so" | grep setuptools)))
export LD_LIBRARY_PATH="${TORCH_PATH}:${AESTREAM_PATH}"
