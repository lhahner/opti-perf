#include <CL/cl.h>
#include <benchmark/benchmark.h>

#include "benchmark/benchmark_trainer.h"
#include "benchmark/workloads/generalmatrixmultiplication/gemm.h"
#include "optimization/adam_optimizer.h"
#include "optimization/adam_optimizer_cl.h"
#include "util/device_param_view.h"
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

	std::cout << "Workload Profile:\n"
		<< "Workload-name: " << gemm.workloadName << ", "
		<< "Workload-type: " << gemm.workloadType << ", "
		<< "Workload-size: " << gemm.dimensions_ << "\n";

	AdamOptimizerCl adam;
	gemm.initializeInput();

	auto* wrapper = DevicePlatformWrapperOpenCL::getInstance();
	wrapper->setup();

	cl_context ctx = wrapper->getClContext();
	cl_command_queue queue = wrapper->getClCommandQueueForDevice();

	cl_program program = wrapper->createProgram(
			ctx,
			wrapper->getDeviceId(),
			"../../kernels/adam_optimizer.cl"
			);

	cl_int err = CL_SUCCESS;
	cl_kernel kernel = clCreateKernel(program, "adam", &err);
	if (err != CL_SUCCESS) {
		std::cerr << "clCreateKernel(adam) failed: " << err << "\n";
		return;
	}

	const size_t local_size = 256;
	adam.configure(ctx, queue, kernel, 1e-3f, 0.9f, 0.999f, 1e-8f, 256);  
	// Warm-up / correctness loop
	for (int t = 1; t <= 10; ++t) {
		gemm.runForward();
		adam.step(gemm.parameters(), t);
		benchmark::DoNotOptimize(gemm.computeLoss());
		std::cout << "[INFO] Computed loss: " << gemm.computeLoss() << std::endl;
	}
	for (auto _ : state) {
		BenchmarkTrainer::runOptimizerWithWorkload(gemm, adam, iters);
	}

	// Clean up (ideally use RAII instead of raw new)
	clReleaseKernel(kernel);
	clReleaseProgram(program);
}
BENCHMARK(BM_GEMM_Adam_cl)->Arg(100);

