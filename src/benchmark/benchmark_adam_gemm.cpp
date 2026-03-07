#pragma once
#include <CL/cl.h>
#include <benchmark/benchmark.h>
#include <chrono>
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

float lr = 1e-3f;
float beta1 = 0.9f;
float beta2 = 0.999f;
float eps = 1e-8f;
int m = 2024;
int k = 2024;
int n = 256;
int batch_size = 10;

static void BM_GEMM_Adam(benchmark::State &state)
{
	BenchmarkData *benchmarkData = new BenchmarkData(
		nullptr,
		(char *)"None",
		(char *)"GEMM_Adam_CPU",
		nullptr,
		(char *)"CPU",
		batch_size,
		m * k * n,
		(char *)"Adam",
		lr,
		beta1,
		beta2,
		eps,
		0.0f,
		0,
		0.0f);

	std::cout << toString(Marker::INFO) << "Running workload for CPU" << std::endl;
	int iters = static_cast<int>(state.range(0));

	GEMM gemm({m, k, n});
	AdamOptimizer adam(lr, beta1, beta2, eps);

	gemm.initializeInput();
	for (int t = 1; t <= batch_size; ++t)
	{
		benchmarkData->setBatchIndex(t);
		gemm.runForward();
		adam.step(benchmarkData, gemm.parameters(), t);
		benchmark::DoNotOptimize(gemm.computeLoss());
		auto loss = gemm.computeLoss();
		benchmarkData->setLoss(loss.second);
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
BENCHMARK(BM_GEMM_Adam)->Arg(100)->Iterations(1);

static void BM_GEMM_Adam_cl(benchmark::State &state)
{
	BenchmarkData *benchmarkData = new BenchmarkData(
		nullptr,
		(char *)"OpenCL",
		(char *)"GEMM_Adam_CPU",
		nullptr,
		(char *)"GPU",
		batch_size,
		m * k * n,
		(char *)"Adam",
		lr,
		beta1,
		beta2,
		eps,
		0.0f,
		0,
		0.0f
	);

	std::cout << "Running workload for OpenCL" << std::endl;
	int iters = static_cast<int>(state.range(0));

	GEMM gemm({m, k, n});

	AdamOptimizerCl adam(lr, beta1, beta2, eps);
	gemm.initializeInput();

	auto *wrapper = DevicePlatformWrapperOpenCL::getInstance();
	int setupSucess = wrapper->setup();
	if (setupSucess != SETUP_SUCCESS)
	{
		std::cerr << toString(Marker::ERROR) << "Setup initalization failed." << std::endl;
		return;
	}

	cl_context ctx = wrapper->getClContext();
	cl_command_queue queue = wrapper->getClCommandQueueForDevice();

	static const char kernel_path[] =
		OPTI_PERF_SOURCE_DIR "/kernels/adam_optimizer.cl";

	cl_program program = wrapper->createProgram(
		ctx,
		wrapper->getDeviceId(),
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
	adam.configure(ctx, queue, kernel, 1e-3f, 0.9f, 0.999f, 1e-8f, 256);
	for (int t = 1; t <= 10; ++t)
	{
		benchmarkData->setBatchIndex(t);
		gemm.runForward();
		adam.step(benchmarkData, gemm.parameters(), t);
		benchmark::DoNotOptimize(gemm.computeLoss());
		auto loss = gemm.computeLoss();
		benchmarkData->setLoss(loss.second);
		std::cout << toString(Marker::INFO) << "Computed loss h: " << loss.first << ", " << loss.second << std::endl;
	}
	for (auto _ : state)
	{
		BenchmarkTrainer::runOptimizerWithWorkload(gemm, adam, iters);
	}
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	delete benchmarkData;
}
BENCHMARK(BM_GEMM_Adam_cl)->Arg(100)->Iterations(1);

static void BM_GEMM_Adam_cuda(benchmark::State &state)
{
	BenchmarkData *benchmarkData = new BenchmarkData(
		nullptr,
		(char *)"CUDA",
		(char *)"GEMM_Adam_CUDA",
		nullptr,
		(char *)"GPU",
		batch_size,
		m * k * n,
		(char *)"Adam",
		lr,
		beta1,
		beta2,
		eps,
		0.0f,
		0,
		0.0f
	);

	std::cout << toString(Marker::INFO) << "Running workload for CUDA" << std::endl;
	int iters = static_cast<int>(state.range(0));

	GEMM gemm({m, k, n});
	AdamOptimizerCu adam(lr, beta1, beta2, eps);

	gemm.initializeInput();
	for (int t = 1; t <= 10; ++t)
	{
		gemm.runForward();
		adam.step(benchmarkData, gemm.parameters(), t);
		benchmark::DoNotOptimize(gemm.computeLoss());
		auto loss = gemm.computeLoss();
		benchmarkData->setLoss(loss.second);
		std::cout << toString(Marker::INFO) << "Computed loss: " << loss.first << ", " << loss.second << std::endl;
	}

	for (auto _ : state)
	{
		auto loss = gemm.computeLoss();
		benchmark::DoNotOptimize(loss);
		std::cout << toString(Marker::INFO) << "Computed loss: (" << loss.first << ", " << loss.second << ")\n";
	}
}
BENCHMARK(BM_GEMM_Adam_cuda)->Arg(100)->Iterations(1);
