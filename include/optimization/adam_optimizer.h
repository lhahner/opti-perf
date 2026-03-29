// optimization/adam_optimizer.h
#pragma once
#include "optimization/optimizer.h"
#include "util/host_param_view.h"
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cmath>
#include <CL/cl.h>
#include <chrono>
#include <iostream>
#include "util/benchmark_data.h"
#include "util/logger.h"
#include <ctime>

/**
 * @brief Implements the Adam optimization algorithm on host-side parameter buffers.
 */
class AdamOptimizer : public Optimizer {
public:
    /**
     * @brief Constructs the Adam optimizer with the provided hyperparameters.
     * @param lr Learning rate.
     * @param beta1 Exponential decay factor for the first moment estimate.
     * @param beta2 Exponential decay factor for the second moment estimate.
     * @param eps Numerical stability epsilon.
     */
    AdamOptimizer(float lr, float beta1, float beta2, float eps)
        : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {}

    /**
     * @brief Performs one Adam update step on the provided host parameters.
     * @param benchmarkData Benchmark row updated with execution metadata.
     * @param params Host-side parameter views to optimize.
     * @param step_index Iteration index of the optimization step.
     */
    void step(BenchmarkData *benchmarkData, const std::vector<HostParamView>& params, int step_index);

private:
   /**
    * @brief Stores the Adam moment vectors for one parameter buffer.
    */
   struct State {
        std::vector<float> m;
        std::vector<float> v;
    }; 
    float lr_, beta1_, beta2_, eps_;
    std::unordered_map<const float*, State> states_;

    /**
     * @brief Returns the cached optimizer state for a parameter buffer.
     * @param p Host parameter view used as the cache key.
     * @return Reference to the state associated with the parameter buffer.
     */
    State& state_for_(const HostParamView& p);
    Logger logger;
};
