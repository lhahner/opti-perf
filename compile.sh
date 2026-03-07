# rm -rf build
CUDA_ARCH="${CUDA_ARCH:-75}"
cmake -S . -B build   -DCMAKE_BUILD_TYPE=Release   -DBUILD_TESTS=OFF   -DLIBTORCH_ROOT=$PWD/lib/libtorch   -DCUDAToolkit_ROOT=/usr/local/cuda   -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda   -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build build -j
./build/app 
