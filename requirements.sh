# Install Google Test Unit-Test Framework
sudo apt install libgtest-dev
cd /usr/src/gtest
sudo cmake .
sudo make
sudo cp lib/*.a /usr/lib

# verify with "cat /usr/include/gtest/gtest.h"
wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.9.1%2Bcu126.zip  
unzip libtorch-shared-with-deps-latest.zip

# Google Benchmark
mkdir -p lib
git clone https://github.com/google/benchmark.git lib/benchmark

# OpenCL
sudo apt update
sudo apt install -y \
  opencl-headers \
  ocl-icd-opencl-dev \
  clinfo

# CUDA
sudo apt install -y gcc-11 g++-11
udo apt install -y   cuda-nvcc-11-8   cuda-cudart-dev-11-8   cuda-compiler-11-8   cuda-libraries-dev-11-8

