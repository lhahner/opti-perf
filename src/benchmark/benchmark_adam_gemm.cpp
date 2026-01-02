#include <CL/cl.h>
#include <benchmark/benchmark.h>

#include "../../include/benchmark/benchmark_trainer.h"
#include "benchmark/workloads/generalmatrixmultiplication/gemm.h"
#include "optimization/adam_optimizer.h"
#include "optimization/adam_optimizer_cl.h"
#include "util/device_platform_wrapper_opencl.h"

static void BM_GEMM_Adam(benchmark::State& state) {
	int iters = static_cast<int>(state.range(0));

	GEMM gemm({10024, 10024, 256});

	std::cout << "Workload Profile: " << "\n"
		<< "Workload-name: " << gemm.workloadName << ", "
		<< "Workload-type: " << gemm.workloadType << " , "
		<< "Workload-size: " << gemm.dimensions_ << std::endl;

	AdamOptimizer adam(1e-3f, 0.9f, 0.999f, 1e-8f);

	gemm.initializeInput();
	for (int t = 1; t <= 10; ++t) {
		gemm.runForward();
		adam.step(gemm.parameters(), t);
		benchmark::DoNotOptimize(gemm.computeLoss());
		std::cout << "[INFO] Computed loss: " << gemm.computeLoss() << std::endl;
	}

	for (auto _ : state) {
		BenchmarkTrainer::runOptimizerWithWorkload(gemm, adam, iters);
	}
}
BENCHMARK(BM_GEMM_Adam)->Arg(100);

static void BM_GEMM_Adam_cl(benchmark::State& state)
{
	int iters = static_cast<int>(state.range(0));

	GEMM gemm({10024, 10024, 256});

	std::cout << "Workload Profile: " << "\n"
		<< "Workload-name: " << gemm.workloadName << ", "
		<< "Workload-type: " << gemm.workloadType << " , "
		<< "Workload-size: " << gemm.dimensions_ << "\n";

	AdamOptimizerCl* adam = new AdamOptimizerCl;	
	gemm.initializeInput();
	DevicePlatformWrapperOpenCL* wrapper = (DevicePlatformWrapperOpenCL*)DevicePlatformWrapperOpenCL::getInstance();

	wrapper->setup();
	cl_command_queue queue = wrapper->getClCommandQueueForDevice();
	cl_program program = wrapper->createProgram(
			wrapper->getClContext(), 
			wrapper->getDeviceId(), 
			"../../kernels/adam_optimizer.cl");
	cl_kernel kernel = clCreateKernel(program, "adam", NULL);
	for (int t = 1; t <= 10; ++t) {
		gemm.runForward();
		adam->step(queue, kernel, adam->transferToBuffer(gemm.parameters()), t, 
			   1e-3f, 0.9f, 0.999f, 1e-8f, 10);
		benchmark::DoNotOptimize(gemm.computeLoss());
		std::cout << "[INFO] Computed loss: " << gemm.computeLoss() << "\n";
	}
	for (auto _ : state) {
		BenchmarkTrainer::runOptimizerWithWorkload(gemm, adam, iters);
	}
}	
BENCHMARK(BM_GEMM_Adam_cl)->Arg(100);
