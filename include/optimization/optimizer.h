#pragma once
#include <vector>
#include "util/host_param_view.h"
#include "util/markers.h"
#include "util/logger.h"
#include <ctime>

/**
 * @brief Defines the common interface for optimizer implementations.
 */
class Optimizer {
public:
    /**
     * @brief Destroys the optimizer instance.
     */
    virtual ~Optimizer() = default;

    /**
     * @brief Executes one optimization step for the provided parameters.
     * @param benchmarkData Benchmark row updated with timing and execution metadata.
     * @param params Host-side parameter views to optimize.
     * @param step_index Iteration index of the optimization step.
     */
    virtual void step(BenchmarkData *benchmarkData, const std::vector<HostParamView>& params, int step_index) = 0;
};
