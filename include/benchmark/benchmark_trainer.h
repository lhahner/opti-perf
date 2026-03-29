#ifndef INCLUDE_BENCHMARK_BENCHMARK_TRAINER_H_
#define INCLUDE_BENCHMARK_BENCHMARK_TRAINER_H_

#include "optimization/optimizer.h"
#include "optimization/adam_optimizer_cl.h"
#include <benchmark/benchmark.h>
#include <vector>
#include <functional>   // std::reference_wrapper, std::ref
#include "benchmark/workloads/workload.h"
#include "../optimization/adam_optimizer.h"

/**
 * @brief Coordinates the execution of workloads together with optimizers.
 */
class BenchmarkTrainer {
public:
    /**
     * @brief Constructs a trainer for a single workload and optimizer pair.
     * @param workload Workload to execute.
     * @param optimizer Optimizer applied to the workload parameters.
     */
    BenchmarkTrainer(Workload& workload, Optimizer& optimizer);

    /**
     * @brief Constructs a trainer for multiple workloads and optimizers.
     * @param workloads Workloads managed by the trainer.
     * @param optimizers Optimizers managed by the trainer.
     */
    BenchmarkTrainer(const std::vector<std::reference_wrapper<Workload>>& workloads,
                     const std::vector<std::reference_wrapper<Optimizer>>& optimizers);

    /**
     * @brief Runs the configured workloads with the available optimizers.
     */
    void runWorkloads();

    /**
     * @brief Runs the configured workloads with the OpenCL optimizer path.
     */
    void runClWorkloads();

    /**
     * @brief Executes an optimizer repeatedly on a single workload.
     * @param workload Workload to execute.
     * @param optimizer Optimizer applied to the workload parameters.
     * @param iters Number of optimization iterations to run.
     */
    static void runOptimizerWithWorkload(Workload& workload, 
		    Optimizer& optimizer, 
		    int iters,
		    BenchmarkData *benchmarkData = nullptr);
    
    /**
     * @brief Returns the workloads managed by the trainer.
     * @return Stored workload references.
     */
    const std::vector<std::reference_wrapper<Workload>>& getWorkloads() const { return workloads_; }

    /**
     * @brief Returns the optimizers managed by the trainer.
     * @return Stored optimizer references.
     */
    const std::vector<std::reference_wrapper<Optimizer>>& getOptimizers() const { return optimizers_; }

private:
    std::vector<std::reference_wrapper<Workload>> workloads_;
    std::vector<std::reference_wrapper<Optimizer>> optimizers_;
};

#endif
