yum install -y zlib-devel ninja-build libgusb-devel wget

# Install libtorch
cd /
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch.zip

# Install libcaer
cd /
wget https://gitlab.com/inivation/dv/libcaer/-/archive/3.3.14/libcaer-3.3.14.tar.gz 
tar xf libcaer-3.3.14.tar.gz
cd libcaer-3.3.14
cmake -DCMAKE_INSTALL_PREFIX=/usr .
make -j 4 && make install
