#!/usr/bin/env bash

# System dependencies
yum install -y zlib-devel ninja-build libgusb-devel

# Ensure we're in the root directory to align paths with publish.yml
cd /root

# Install libtorch
curl -L -s -m 100 https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip > libtorch.zip
# We use Python to Unzip because
#  - Modern linux systems uses PIDs above 65k
#  - Old versions of unzip do not cope well with high PIDs
/usr/local/bin/python3.9 -c '
import zipfile
with zipfile.ZipFile("libtorch.zip", "r") as zip_ref:
  zip_ref.extractall(".")
'

echo "Libtorch extracted"
ls -alh /root/libtorch
  
#unzip -q libtorch.zip

# Install libcaer
curl -L -s -m 100 https://gitlab.com/inivation/dv/libcaer/-/archive/3.3.14/libcaer-3.3.14.tar.gz | tar zxf -
cd libcaer-3.3.14
cmake -DCMAKE_INSTALL_PREFIX=/usr .
make -j 4 && make install
cd ..
