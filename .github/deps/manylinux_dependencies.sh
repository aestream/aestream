yum install -y zlib-devel ninja-build libgusb-devel wget

# Install libtorch
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip -O libtorch.zip
unzip libtorch.zip

# Install libcaer
wget https://gitlab.com/inivation/dv/libcaer/-/archive/3.3.14/libcaer-3.3.14.tar.gz -O libcaer.tar.gz
tar xf libcaer.tar.gz
cd libcaer-3.3.14
cmake -DCMAKE_INSTALL_PREFIX=/usr .
make -j 4 && make install
cd ..
