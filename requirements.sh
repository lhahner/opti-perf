# Install Google Test Unit-Test Framework
sudo apt install libgtest-dev
cd /usr/src/gtest
sudo cmake .
sudo make
sudo cp lib/*.a /usr/lib

# verify with "cat /usr/include/gtest/gtest.h"
