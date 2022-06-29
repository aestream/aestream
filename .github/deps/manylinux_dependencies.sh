yum install -y zlib-devel ninja-build libgusb-devel

# Install libtorch
cd /
curl -L https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip > libtorch.zip
unzip libtorch.zip
# export CMAKE_PREFIX_PATH=/libtorch:/libtorch/lib:${CMAKE_PREFIX_PATH}
# export LD_LIBRARY_PATH=/libtorch:/libtorch/lib:${LD_LIBRARY_PATH}
# export PATH=/libtorch/:/libtorch/lib:${PATH}

# Install libcaer
cd /
curl -L https://gitlab.com/inivation/dv/libcaer/-/archive/3.3.14/libcaer-3.3.14.tar.gz | tar zxf -
cd libcaer-3.3.14
cmake -DCMAKE_INSTALL_PREFIX=/usr .
make -j 4 && make install
