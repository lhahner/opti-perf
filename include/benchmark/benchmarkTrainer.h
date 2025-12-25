#ifndef INCLUDE_BENCHMARK_BENCHMARKTRAINER_H_
#define INCLUDE_BENCHMARK_BENCHMARKTRAINER_H_

#include "optimization/optimizer.h"
#include <benchmark/benchmark.h>
#include <vector>
#include <functional>   // std::reference_wrapper, std::ref
#include "benchmark/workloads/machinelearning/mnist.h"
#include "optimization/stochastic_gradient_descent.h"

class BenchmarkTrainer {
public:
    BenchmarkTrainer(Workload& workload, Optimizer& optimizer);

    BenchmarkTrainer(const std::vector<std::reference_wrapper<Workload>>& workloads,
                     const std::vector<std::reference_wrapper<Optimizer>>& optimizers);

    void runWorkloads();
    static void runOptimizerWithWorkload(Workload &workload, const std::vector<std::reference_wrapper<Optimizer>>& optimizers);

    const std::vector<std::reference_wrapper<Workload>>& getWorkloads() const { return workloads_; }
    const std::vector<std::reference_wrapper<Optimizer>>& getOptimizers() const { return optimizers_; }

private:
    std::vector<std::reference_wrapper<Workload>> workloads_;
    std::vector<std::reference_wrapper<Optimizer>> optimizers_;
};

#endif
