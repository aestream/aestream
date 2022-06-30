# System dependencies
yum install -y zlib-devel ninja-build libgusb-devel

echo `pwd`

# Ensure we're in the root
cd /root

# Install libtorch
curl -L -s -m 100 https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip > libtorch.zip
ls -alh .
unzip -q libtorch.zip

# Install libcaer
curl -L -s -m 100 https://gitlab.com/inivation/dv/libcaer/-/archive/3.3.14/libcaer-3.3.14.tar.gz | tar zxf -
cd libcaer-3.3.14
cmake -DCMAKE_INSTALL_PREFIX=/usr .
make -j 4 && make install
cd ..
