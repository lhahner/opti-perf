# Install Google Test Unit-Test Framework
sudo apt install libgtest-dev
cd /usr/src/gtest
sudo cmake .
sudo make
sudo cp lib/*.a /usr/lib

# verify with "cat /usr/include/gtest/gtest.h"
wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.9.1%2Bcu126.zip  
unzip libtorch-shared-with-deps-latest.zip
