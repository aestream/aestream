yum install -y zlib-devel ninja-build libgusb-devel

# Install libtorch
cd /
curl -L https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip > libtorch.zip
unzip libtorch.zip

# Install libcaer
cd /
curl -L https://gitlab.com/inivation/dv/libcaer/-/archive/3.3.14/libcaer-3.3.14.tar.gz | tar zxf -
cd libcaer-3.3.14
cmake -DCMAKE_INSTALL_PREFIX=/usr .
make -j 4 && make install
