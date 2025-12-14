cmake_minimum_required(VERSION 3.18)
project(app LANGUAGES CXX)

# -----------------------------
# Build configuration
# -----------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type if not specified: Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------
# LibTorch configuration
# -----------------------------
# Mirrors your Makefile: LIBTORCH_ROOT := $(CURDIR)/lib/libtorch
set(LIBTORCH_ROOT "${CMAKE_SOURCE_DIR}/lib/libtorch")

# Tell CMake where TorchConfig.cmake is
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")

find_package(Torch REQUIRED)

# ABI flag must match how libtorch was built; keep your Makefile default
# If you later find you need ABI=0, change this to 0.
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

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

# Your Makefile has -fopenmp
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(app PRIVATE OpenMP::OpenMP_CXX)
else()
  # If OpenMP module isn't available, you can still try with the flag
  target_compile_options(app PRIVATE -fopenmp)
  target_link_options(app PRIVATE -fopenmp)
endif()

# Warnings like Makefile
target_compile_options(app PRIVATE -Wall -Wextra -Wpedantic)

# Link LibTorch the supported way
target_link_libraries(app PRIVATE "${TORCH_LIBRARIES}")

# Optional: ensure rpath so you don't need LD_LIBRARY_PATH at runtime
# (similar to your Makefile -Wl,-rpath,...)
set_target_properties(app PROPERTIES
  BUILD_RPATH "${LIBTORCH_ROOT}/lib"
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

  # Link with torch too (tests use your code + torch)
  target_link_libraries(tests PRIVATE "${TORCH_LIBRARIES}")

  if(OpenMP_CXX_FOUND)
    target_link_libraries(tests PRIVATE OpenMP::OpenMP_CXX)
  else()
    target_compile_options(tests PRIVATE -fopenmp)
    target_link_options(tests PRIVATE -fopenmp)
  endif()

  target_compile_options(tests PRIVATE -Wall -Wextra -Wpedantic)
  set_target_properties(tests PROPERTIES BUILD_RPATH "${LIBTORCH_ROOT}/lib")

  # --- GoogleTest ---
  # Simplest: use system-installed gtest if available.
  # If your HPC does not have it, see Step 4 below (FetchContent).
  find_package(GTest REQUIRED)
  target_link_libraries(tests PRIVATE GTest::gtest GTest::gtest_main pthread)

  add_test(NAME unit_tests COMMAND tests)
endif()

