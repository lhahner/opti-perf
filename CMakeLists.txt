cmake_minimum_required(VERSION 3.18)
project(opti_perf LANGUAGES CXX CUDA)

add_compile_options(-w)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Default to RTX5000 (sm_75) unless the user overrides via -DCMAKE_CUDA_ARCHITECTURES.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA arch list" FORCE)
endif()
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^[0-9]+$")
	# Some dependencies inject CUDA gencode flags; force-append the requested arch.
	string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}")
endif()

option(ENABLE_YAML "Enable yaml-cpp integration" ON)
option(USE_SYSTEM_YAMLCPP "Use system-provided yaml-cpp" OFF)


if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- OpenMP ----
find_package(OpenMP)

# ---- yaml-cpp ----
if(ENABLE_YAML)
	if(USE_SYSTEM_YAMLCPP)
		find_package(yaml-cpp REQUIRED)
	else()
		set(YAMLCPP_ROOT "${CMAKE_SOURCE_DIR}/lib/yaml-cpp" CACHE PATH "Path to yaml-cpp source")
		if(EXISTS "${YAMLCPP_ROOT}/CMakeLists.txt")
			set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
			set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
			add_subdirectory("${YAMLCPP_ROOT}" EXCLUDE_FROM_ALL)
		else()
			message(FATAL_ERROR "yaml-cpp not found. Set YAMLCPP_ROOT or enable USE_SYSTEM_YAMLCPP.")
		endif()
	endif()

	# Normalize target name for linking.
	if(TARGET yaml-cpp::yaml-cpp)
		set(YAMLCPP_TARGET yaml-cpp::yaml-cpp)
	elseif(TARGET yaml-cpp)
		set(YAMLCPP_TARGET yaml-cpp)
	else()
		message(FATAL_ERROR "yaml-cpp target not available after configuration.")
	endif()
endif()

# ---- CUDA ----
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -lineinfo")

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
	target_compile_definitions(${tgt} PRIVATE OPTI_PERF_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	target_link_libraries(${tgt})
	target_enable_openmp(${tgt})
	target_enable_opencl(${tgt})
	if(ENABLE_YAML)
		target_link_libraries(${tgt} PRIVATE ${YAMLCPP_TARGET})
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
target_sources(workloads PRIVATE
	${CMAKE_SOURCE_DIR}/src/util/device_platform_handler_cuda.cpp
)
target_link_libraries(workloads)
target_enable_openmp(workloads)
target_enable_opencl(workloads)
if(ENABLE_YAML)
	target_link_libraries(workloads PRIVATE ${YAMLCPP_TARGET})
endif()
target_enable_cuda(workloads PUBLIC)

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
	# Build the vendored benchmark
	add_subdirectory(${CMAKE_SOURCE_DIR}/lib/benchmark EXCLUDE_FROM_ALL)
endif()

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
target_compile_definitions(app PRIVATE OPTI_PERF_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(app PRIVATE workloads)
# app.cpp calls register_adam_gemm_benchmarks; add its implementation explicitly.
target_sources(app PRIVATE
	${CMAKE_SOURCE_DIR}/src/benchmark/benchmark_adam_gemm.cpp
	${CMAKE_SOURCE_DIR}/src/benchmark/benchmark_adam_training.cpp
	${CMAKE_SOURCE_DIR}/src/benchmark/benchmark_trainer.cpp
)
if(BUILD_BENCHMARKS)
	target_link_libraries(app PRIVATE benchmark::benchmark)
else()
	message(FATAL_ERROR "BUILD_BENCHMARKS=OFF is not supported because app depends on Google Benchmark.")
endif()

# -----------------------------
# Bench (Google Benchmark)
# -----------------------------
if(BUILD_BENCHMARKS)
	file(GLOB_RECURSE BENCH_SOURCES CONFIGURE_DEPENDS
		"${CMAKE_SOURCE_DIR}/src/benchmark/*.cpp"
	)
	# Exclude workload implementations from bench sources (they come from workloads lib)
	list(FILTER BENCH_SOURCES EXCLUDE REGEX ".*/src/benchmark/workloads/.*\\.cpp$")
	add_executable(bench
		${BENCH_SOURCES}

		# Explicitly add the implementation files that bench needs
		${CMAKE_SOURCE_DIR}/src/util/config_reader.cpp
		${CMAKE_SOURCE_DIR}/src/util/logger.cpp
		${CMAKE_SOURCE_DIR}/src/util/device_platform_wrapper_opencl.cpp
		${CMAKE_SOURCE_DIR}/src/optimization/adam_optimizer.cpp
		${CMAKE_SOURCE_DIR}/src/optimization/adam_optimizer_cl.cpp
	)
	target_apply_common(bench)
	target_enable_cuda(bench)
	target_compile_definitions(bench PRIVATE OPTI_PERF_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
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
