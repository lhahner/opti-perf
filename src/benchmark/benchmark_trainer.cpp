#include "../../include/benchmark/benchmark_trainer.h"

BenchmarkTrainer::BenchmarkTrainer(Workload& workload, Optimizer& optimizer) {
    workloads_.push_back(std::ref(workload));
    optimizers_.push_back(std::ref(optimizer));
}

BenchmarkTrainer::BenchmarkTrainer(
    const std::vector<std::reference_wrapper<Workload>>& workloads,
    const std::vector<std::reference_wrapper<Optimizer>>& optimizers
) : workloads_(workloads), optimizers_(optimizers) {}

void BenchmarkTrainer::runWorkloads() {
	for (auto workload : getWorkloads()) {
		this->runOptimizerWithWorkload(workload, BenchmarkTrainer::getOptimizers().at(0), 10);
	}
}

void BenchmarkTrainer::runOptimizerWithWorkload(
    Workload& workload,
    Optimizer& optimizer,
    int iters)
{
    // Important: ensure identical starting state each benchmark iteration
    //workload.initializeInput(); // or workload.reset()

    for (int t = 1; t <= iters; ++t) {
        workload.runForward();
        optimizer.step(workload.parameters(), t);
        benchmark::DoNotOptimize(workload.computeLoss());
    }
}



