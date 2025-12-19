# rm -rf build
cmake -S . -B build   -DCMAKE_BUILD_TYPE=Release   -DBUILD_TESTS=OFF   -DLIBTORCH_ROOT=$PWD/lib/libtorch   -DCUDAToolkit_ROOT=/usr/local/cuda   -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
cmake --build build -j
./build/app 

