cmake_minimum_required(VERSION 3.18)
project(opti_perf LANGUAGES CXX)

# -----------------------------
# Build configuration
# -----------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------
# LibTorch configuration
# -----------------------------
# Allow override from CLI:
#   cmake -S . -B build -DLIBTORCH_ROOT=/path/to/libtorch-cuda
set(LIBTORCH_ROOT "${CMAKE_SOURCE_DIR}/lib/libtorch" CACHE PATH "Path to LibTorch root")

# TorchConfig.cmake lives under <LIBTORCH_ROOT>/share/cmake/Torch typically,
# but adding LIBTORCH_ROOT to CMAKE_PREFIX_PATH is enough.
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")

find_package(Torch REQUIRED)

message(STATUS "Using LibTorch from: ${LIBTORCH_ROOT}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")

# Fail-fast / warn if this is a CPU-only LibTorch
if(EXISTS "${LIBTORCH_ROOT}/lib/libtorch_cuda.so")
  message(STATUS "CUDA-enabled LibTorch detected (libtorch_cuda.so found).")
else()
  message(WARNING "CPU-only LibTorch detected (libtorch_cuda.so NOT found). "
                  "torch::cuda::is_available() will be false even on GPU nodes. "
                  "Install/use a CUDA LibTorch and reconfigure with -DLIBTORCH_ROOT=...")
endif()

# IMPORTANT:
# Do NOT force _GLIBCXX_USE_CXX11_ABI unless you are sure it matches your LibTorch.
# If you need it, set it via CMake configure:
#   -DGLIBCXX_USE_CXX11_ABI=1  (or 0)
option(GLIBCXX_USE_CXX11_ABI "Set _GLIBCXX_USE_CXX11_ABI (must match LibTorch)" OFF)
if(GLIBCXX_USE_CXX11_ABI)
  # Default to 1; override with -DGLIBCXX_USE_CXX11_ABI_VALUE=0 if needed
  set(GLIBCXX_USE_CXX11_ABI_VALUE 1 CACHE STRING "Value for _GLIBCXX_USE_CXX11_ABI (0 or 1)")
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI_VALUE})
  message(STATUS "Forcing _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI_VALUE}")
endif()

# -----------------------------
# Sources
# -----------------------------
file(GLOB_RECURSE APP_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

add_executable(app ${APP_SOURCES})

target_include_directories(app PRIVATE
  "${CMAKE_SOURCE_DIR}/include"
)

# Warnings like Makefile
target_compile_options(app PRIVATE -Wall -Wextra -Wpedantic)

# Apply Torch compile flags to this target (important)
target_compile_options(app PRIVATE ${TORCH_CXX_FLAGS})

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(app PRIVATE OpenMP::OpenMP_CXX)
else()
  target_compile_options(app PRIVATE -fopenmp)
  target_link_options(app PRIVATE -fopenmp)
endif()

# Link LibTorch
target_link_libraries(app PRIVATE "${TORCH_LIBRARIES}")

# RPATH so runtime can find libtorch without setting LD_LIBRARY_PATH
set_target_properties(app PROPERTIES
  BUILD_RPATH "${LIBTORCH_ROOT}/lib"
  INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${LIBTORCH_ROOT}/lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# -----------------------------
# Tests (GoogleTest)
# -----------------------------
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
  enable_testing()

  file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/tests/*.cpp"
  )

  add_executable(tests ${TEST_SOURCES})

  target_include_directories(tests PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
  )

  target_compile_options(tests PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(tests PRIVATE ${TORCH_CXX_FLAGS})

  if(OpenMP_CXX_FOUND)
    target_link_libraries(tests PRIVATE OpenMP::OpenMP_CXX)
  else()
    target_compile_options(tests PRIVATE -fopenmp)
    target_link_options(tests PRIVATE -fopenmp)
  endif()

  target_link_libraries(tests PRIVATE "${TORCH_LIBRARIES}")

  set_target_properties(tests PROPERTIES
    BUILD_RPATH "${LIBTORCH_ROOT}/lib"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${LIBTORCH_ROOT}/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )

  find_package(GTest REQUIRED)
  target_link_libraries(tests PRIVATE GTest::gtest GTest::gtest_main pthread)

  add_test(NAME unit_tests COMMAND tests)
endif()

