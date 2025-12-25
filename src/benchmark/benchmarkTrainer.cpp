#include "benchmark/benchmarkTrainer.h"

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
		this->runOptimizerWithWorkload(workload, BenchmarkTrainer::getOptimizers());
	}
}

void BenchmarkTrainer::runOptimizerWithWorkload(Workload& workload, const std::vector<std::reference_wrapper<Optimizer>>& optimizers) {
	for (auto optimizer : optimizers) {
				workload.runForward();
				workload.computeLoss();
	}
}
//BENCHMARK(BenchmarkTrainer::runOptimizerWithWorkload);


