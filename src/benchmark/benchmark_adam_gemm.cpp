#include <benchmark/benchmark.h>

#include "../../include/benchmark/benchmark_trainer.h"
#include "benchmark/workloads/generalmatrixmultiplication/gemm.h"
#include "optimization/adam_optimizer.h"

static void BM_GEMM_Adam(benchmark::State& state) {
    // state.range(0) = iters per benchmark iteration
    int iters = static_cast<int>(state.range(0));

    // Example dimensions: M,K,N
    GEMM gemm({10024, 10024, 256});

    std::cout << "Workload Profile: " << "\n"
    		  << "Workload-name: " << gemm.workloadName << ", "
			  << "Workload-type: " << gemm.workloadType << " , "
			  << "Workload-size: " << gemm.dimensions_ << "\n";

    AdamOptimizer adam(/*lr*/1e-3f, /*b1*/0.9f, /*b2*/0.999f, /*eps*/1e-8f);

    // Optional warmup (not counted)
    gemm.initializeInput();
    for (int t = 1; t <= 10; ++t) {
        gemm.runForward();
        adam.step(gemm.parameters(), t);
        benchmark::DoNotOptimize(gemm.computeLoss());
        std::cout << "[INFO] Computed loss: " << gemm.computeLoss() << "\n";
    }

    for (auto _ : state) {
        BenchmarkTrainer::runOptimizerWithWorkload(gemm, adam, iters);
    }
}
BENCHMARK(BM_GEMM_Adam)->Arg(100);
