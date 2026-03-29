#pragma once
#include <CL/cl.h>
#include <benchmark/benchmark.h>
#include <chrono>
#include <stdexcept>
#include <string>

#include "benchmark/benchmark_adam_gemm.h"
#include "benchmark/benchmark_trainer.h"
#include "benchmark/workloads/generalmatrixmultiplication/gemm.h"
#include "optimization/adam_optimizer.h"
#include "optimization/adam_optimizer_cl.h"
#include "optimization/adam_optimizer_cu.h"
#include "util/device_param_view.h"
#include "util/device_platform_wrapper_opencl.h"

#ifndef OPTI_PERF_SOURCE_DIR
#define OPTI_PERF_SOURCE_DIR "."
#endif

namespace {

struct AdamGemmConfig
{
	float learning_rate = 1e-3f;
	float beta1 = 0.9f;
	float beta2 = 0.999f;
	float epsilon = 1e-8f;
	long width = 2024;
	long height = 2024;
	long depth = 256;
	int batch_size = 10;
	std::string framework = "CPU";
};

AdamGemmConfig g_config;

const AdamGemmConfig &config()
{
	return g_config;
}

AdamGemmConfig load_config(ConfigReader &config_reader)
{
	const YAML::Node runtime = config_reader.get_runtime_config();
	const YAML::Node optimizer = config_reader.get_optimizer_config();

	AdamGemmConfig config;
	config.framework = runtime["framework"].as<std::string>();
	config.learning_rate = optimizer["learning_rate"].as<float>();
	config.beta1 = optimizer["beta_1"].as<float>();
	config.beta2 = optimizer["beta_2"].as<float>();
	config.epsilon = optimizer["epsilon"].as<float>();
	config.width = optimizer["dim_m"].as<long>();
	config.height = optimizer["dim_k"].as<long>();
	config.depth = optimizer["dim_n"].as<long>();
	config.batch_size = optimizer["batch_size"].as<int>();
	return config;
}

} // namespace

static void BM_GEMM_Adam(benchmark::State &state)
{
	const AdamGemmConfig &cfg = config();
	BenchmarkData *benchmarkData = new BenchmarkData(
		nullptr,
		"CPU",
		(char *)"None",
		(char *)"GEMM_Adam_CPU",
		nullptr,
		(char *)"CPU",
		cfg.batch_size,
		cfg.width * cfg.height * cfg.depth,
		(char *)"Adam",
		cfg.learning_rate,
		cfg.beta1,
		cfg.beta2,
		cfg.epsilon,
		0.0f,
		0,
		0.0f);

	std::cout << toString(Marker::INFO) << "Running workload for CPU" << std::endl;
	int iters = static_cast<int>(state.range(0));

	GEMM gemm({cfg.width, cfg.height, cfg.depth});
	AdamOptimizer adam(cfg.learning_rate, cfg.beta1, cfg.beta2, cfg.epsilon);

	gemm.initializeInput();
	for (int t = 1; t <= cfg.batch_size; ++t)
	{
		benchmarkData->batch_index = t;
		gemm.runForward();
		adam.step(benchmarkData, gemm.parameters(), t);
		benchmark::DoNotOptimize(gemm.computeLoss());
		auto loss = gemm.computeLoss();
		benchmarkData->loss = loss.second;
		std::cout << toString(Marker::INFO) << "Computed loss: " << loss.first << ", " << loss.second << std::endl;
	}

	for (auto _ : state)
	{
		auto loss = gemm.computeLoss();
		benchmark::DoNotOptimize(loss);
		std::cout << toString(Marker::INFO) << "Computed loss: (" << loss.first << ", " << loss.second << ")\n";
	}
	delete benchmarkData;
}

static void BM_GEMM_Adam_cl(benchmark::State &state)
{
	const AdamGemmConfig &cfg = config();
	BenchmarkData *benchmarkData = new BenchmarkData(
		nullptr,
		nullptr,
		(char *)"OpenCL",
		(char *)"GEMM_Adam_CPU",
		nullptr,
		(char *)"GPU",
		cfg.batch_size,
		cfg.width * cfg.height * cfg.depth,
		(char *)"Adam",
		cfg.learning_rate,
		cfg.beta1,
		cfg.beta2,
		cfg.epsilon,
		0.0f,
		0,
		0.0f
	);

	std::cout << "Running workload for OpenCL" << std::endl;
	int iters = static_cast<int>(state.range(0));

	GEMM gemm({cfg.width, cfg.height, cfg.depth});

	AdamOptimizerCl adam(cfg.learning_rate, cfg.beta1, cfg.beta2, cfg.epsilon);
	gemm.initializeInput();

	auto *wrapper = DevicePlatformWrapperOpenCL::get_instance();
	int setupSucess = wrapper->setup();
	if (setupSucess != SETUP_SUCCESS)
	{
		std::cerr << toString(Marker::ERROR) << "Setup initalization failed." << std::endl;
		return;
	}
	benchmarkData->device_name = wrapper->get_device_name();

	cl_context ctx = wrapper->get_context();
	cl_command_queue queue = wrapper->get_command_queue();

	static const char kernel_path[] =
		OPTI_PERF_SOURCE_DIR "/kernels/adam_optimizer.cl";

	cl_program program = wrapper->create_program(
		ctx,
		wrapper->get_device_id(),
		kernel_path);

	cl_int err = CL_SUCCESS;

	static const char kernelName[] = "adam_update";
	cl_kernel kernel = clCreateKernel(program, kernelName, &err);
	if (err != CL_SUCCESS)
	{
		std::cerr << toString(Marker::ERROR) << "clCreateKernel(adam) failed: " << err << "\n";
		return;
	}

	const size_t local_size = 256;
	(void)local_size;
	adam.configure(ctx, queue, kernel, cfg.learning_rate, cfg.beta1, cfg.beta2, cfg.epsilon, 256);
	for (int t = 1; t <= cfg.batch_size; ++t)
	{
		benchmarkData->batch_index = t;
		gemm.runForward();
		adam.step(benchmarkData, gemm.parameters(), t);
		benchmark::DoNotOptimize(gemm.computeLoss());
		auto loss = gemm.computeLoss();
		benchmarkData->loss = loss.second;
		std::cout << toString(Marker::INFO) << "Computed loss h: " << loss.first << ", " << loss.second << std::endl;
	}
	for (auto _ : state)
	{
		BenchmarkTrainer::runOptimizerWithWorkload(gemm, adam, iters, benchmarkData);
	}
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	delete benchmarkData;
}

static void BM_GEMM_Adam_cuda(benchmark::State &state)
{
	const AdamGemmConfig &cfg = config();
	BenchmarkData *benchmarkData = new BenchmarkData(
		nullptr,
		"CUDA",
		(char *)"CUDA",
		(char *)"GEMM_Adam_CUDA",
		nullptr,
		(char *)"GPU",
		cfg.batch_size,
		cfg.width * cfg.height * cfg.depth,
		(char *)"Adam",
		cfg.learning_rate,
		cfg.beta1,
		cfg.beta2,
		cfg.epsilon,
		0.0f,
		0,
		0.0f
	);

	std::cout << toString(Marker::INFO) << "Running workload for CUDA" << std::endl;
	int iters = static_cast<int>(state.range(0));

	GEMM gemm({cfg.width, cfg.height, cfg.depth});
	AdamOptimizerCu adam(cfg.learning_rate, cfg.beta1, cfg.beta2, cfg.epsilon);

	gemm.initializeInput();
	for (int t = 1; t <= cfg.batch_size; ++t)
	{
		gemm.runForward();
		adam.step(benchmarkData, gemm.parameters(), t);
		benchmark::DoNotOptimize(gemm.computeLoss());
		auto loss = gemm.computeLoss();
		benchmarkData->loss = loss.second;
		std::cout << toString(Marker::INFO) << "Computed loss: " << loss.first << ", " << loss.second << std::endl;
	}

	for (auto _ : state)
	{
		auto loss = gemm.computeLoss();
		benchmark::DoNotOptimize(loss);
		std::cout << toString(Marker::INFO) << "Computed loss: (" << loss.first << ", " << loss.second << ")\n";
	}
}

bool register_adam_gemm_benchmarks(ConfigReader &config_reader)
{
	const YAML::Node runtime = config_reader.get_runtime_config();
	const std::string workload = runtime["workload"].as<std::string>();
	const std::string optimizer = runtime["optimizer"].as<std::string>();

	if (workload != "GEMM" || optimizer != "Adam")
	{
		return false;
	}

	g_config = load_config(config_reader);

	if (g_config.framework == "CPU" || g_config.framework == "None")
	{
		benchmark::RegisterBenchmark("BM_GEMM_Adam", &BM_GEMM_Adam)->Arg(100)->Iterations(1);
		return true;
	}

	if (g_config.framework == "OpenCL")
	{
		benchmark::RegisterBenchmark("BM_GEMM_Adam_cl", &BM_GEMM_Adam_cl)->Arg(100)->Iterations(1);
		return true;
	}

	if (g_config.framework == "CUDA")
	{
		benchmark::RegisterBenchmark("BM_GEMM_Adam_cuda", &BM_GEMM_Adam_cuda)->Arg(100)->Iterations(1);
		return true;
	}

	throw std::runtime_error("Unsupported framework in runtime config: " + g_config.framework);
}
