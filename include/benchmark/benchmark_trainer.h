#ifndef INCLUDE_BENCHMARK_BENCHMARK_TRAINER_H_
#define INCLUDE_BENCHMARK_BENCHMARK_TRAINER_H_

#include "optimization/optimizer.h"
#include "optimization/adam_optimizer_cl.h"
#include <benchmark/benchmark.h>
#include <vector>
#include <functional>   // std::reference_wrapper, std::ref

#include "../optimization/adam_optimizer.h"
#include "benchmark/workloads/machinelearning/mnist.h"

class BenchmarkTrainer {
public:
    BenchmarkTrainer(Workload& workload, Optimizer& optimizer);

    BenchmarkTrainer(const std::vector<std::reference_wrapper<Workload>>& workloads,
                     const std::vector<std::reference_wrapper<Optimizer>>& optimizers);

    void runWorkloads();
    void runClWorkloads();
    static void runOptimizerWithWorkload(Workload& workload, 
		    Optimizer& optimizer, 
		    int iters);
    
    const std::vector<std::reference_wrapper<Workload>>& getWorkloads() const { return workloads_; }
    const std::vector<std::reference_wrapper<Optimizer>>& getOptimizers() const { return optimizers_; }

private:
    std::vector<std::reference_wrapper<Workload>> workloads_;
    std::vector<std::reference_wrapper<Optimizer>> optimizers_;
};

#endif
