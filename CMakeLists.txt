cmake_minimum_required(VERSION 3.18)
project(opti_perf LANGUAGES CXX CUDA)

add_compile_options(-w)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 52)

option(ENABLE_TORCH "Enable LibTorch integration" ON)


if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- LibTorch ----

set(LIBTORCH_ROOT "${CMAKE_SOURCE_DIR}/lib/libtorch" CACHE PATH "Path to LibTorch root")

if(ENABLE_TORCH)
	list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
	# More robust than CMAKE_PREFIX_PATH alone:
	set(Torch_DIR "${LIBTORCH_ROOT}/share/cmake/Torch" CACHE PATH "Torch CMake config dir")
	find_package(Torch REQUIRED)
else()
	message(STATUS "LibTorch disabled (ENABLE_TORCH=OFF)")
endif()

# ---- OpenMP ----
find_package(OpenMP)

# ---- CUDA ----
find_package(CUDAToolkit REQUIRED)

# ---- OpenCL ----
find_package(OpenCL REQUIRED)

function(target_enable_opencl tgt)
	# CMake 3.18+ FindOpenCL typically provides OpenCL::OpenCL
	if(TARGET OpenCL::OpenCL)
		target_link_libraries(${tgt} PRIVATE OpenCL::OpenCL)
	else()
		# Fallback for older/alternate FindOpenCL variants
		target_include_directories(${tgt} PRIVATE ${OpenCL_INCLUDE_DIRS})
		target_link_libraries(${tgt} PRIVATE ${OpenCL_LIBRARIES})
	endif()
endfunction()

function(target_enable_cuda tgt)
	target_link_libraries(${tgt} PRIVATE CUDA::cudart)
	target_compile_definitions(${tgt} PRIVATE HAVE_CUDA=1)
	set_target_properties(${tgt} PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
		CUDA_RESOLVE_DEVICE_SYMBOLS ON
	)
	target_compile_definitions(${tgt} PRIVATE HAVE_CUDA=1)
endfunction()

function(target_enable_openmp tgt)
	if(OpenMP_CXX_FOUND)
		target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
	else()
		target_compile_options(${tgt} PRIVATE -fopenmp)
		target_link_options(${tgt} PRIVATE -fopenmp)
	endif()
endfunction()

function(target_apply_common tgt)
	target_include_directories(${tgt} PRIVATE "${CMAKE_SOURCE_DIR}/include")
	target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic)
	target_link_libraries(${tgt})
	target_enable_openmp(${tgt})
	target_enable_opencl(${tgt})

	if(ENABLE_TORCH)
		set_target_properties(${tgt} PROPERTIES
			BUILD_RPATH "${LIBTORCH_ROOT}/lib"
			INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${LIBTORCH_ROOT}/lib"
			INSTALL_RPATH_USE_LINK_PATH TRUE
		)
	endif()

endfunction()

# -----------------------------
# Workloads library (shared)
# -----------------------------
file(GLOB_RECURSE WORKLOAD_SOURCES CONFIGURE_DEPENDS
	"${CMAKE_SOURCE_DIR}/src/benchmark/workloads/*.cpp"
)
file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/**/*.cu"
)
add_library(workloads STATIC ${WORKLOAD_SOURCES} ${CUDA_SOURCES})
target_include_directories(workloads PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_compile_options(workloads PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(workloads)
target_enable_openmp(workloads)
target_enable_opencl(workloads)
target_enable_cuda(workloads PUBLIC)

# -----------------------------
# App (no benchmark harness sources)
# -----------------------------
file(GLOB_RECURSE APP_SOURCES CONFIGURE_DEPENDS
	"${CMAKE_SOURCE_DIR}/src/*.cpp"
)
# Exclude EVERYTHING in src/benchmark from app
list(FILTER APP_SOURCES EXCLUDE REGEX ".*/src/benchmark/.*\\.cpp$")

add_executable(app ${APP_SOURCES})
target_apply_common(app)
target_enable_cuda(app)
target_link_libraries(app PRIVATE workloads)

# -----------------------------
# Bench (Google Benchmark)
# -----------------------------
option(BUILD_BENCHMARKS "Build google-benchmark benchmarks" ON)

if(BUILD_BENCHMARKS)
	# --- Google Benchmark (vendored) ---
	# These are the options used by google/benchmark's CMake
	set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

	# Older/newer versions vary; these are safe to set:
	set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
	set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
	# Extract _GLIBCXX_USE_CXX11_ABI from TORCH_CXX_FLAGS, if present, and apply globally
	if(ENABLE_TORCH)
		string(REGEX MATCH "-D_GLIBCXX_USE_CXX11_ABI=([01])" _abi_match "${TORCH_CXX_FLAGS}")
	endif()

	if(_abi_match)
		set(_abi_value "${CMAKE_MATCH_1}")
		add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${_abi_value})
		message(STATUS "Forcing _GLIBCXX_USE_CXX11_ABI=${_abi_value} for all targets (incl. vendored benchmark)")
	endif()
	# Build the vendored benchmark
	add_subdirectory(${CMAKE_SOURCE_DIR}/lib/benchmark EXCLUDE_FROM_ALL)

	file(GLOB_RECURSE BENCH_SOURCES CONFIGURE_DEPENDS
		"${CMAKE_SOURCE_DIR}/src/benchmark/*.cpp"
	)
	# Exclude workload implementations from bench sources (they come from workloads lib)
	list(FILTER BENCH_SOURCES EXCLUDE REGEX ".*/src/benchmark/workloads/.*\\.cpp$")
	add_executable(bench
		${BENCH_SOURCES}

		# Explicitly add the implementation files that bench needs
		${CMAKE_SOURCE_DIR}/src/util/device_platform_wrapper_opencl.cpp
		${CMAKE_SOURCE_DIR}/src/optimization/adam_optimizer.cpp
		${CMAKE_SOURCE_DIR}/src/optimization/adam_optimizer_cl.cpp
	)
	target_apply_common(bench)
	target_link_libraries(bench PRIVATE workloads benchmark::benchmark benchmark::benchmark_main)

endif()

# -----------------------------
# Tests (GoogleTest)
# -----------------------------
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
	enable_testing()
	file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/tests/*.cpp")
	add_executable(tests ${TEST_SOURCES})
	target_apply_common(tests)

	find_package(GTest REQUIRED)
	target_link_libraries(tests PRIVATE GTest::gtest GTest::gtest_main pthread workloads)

	add_test(NAME unit_tests COMMAND tests)
endif()
