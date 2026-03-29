// optimization/adam_optimizer.h
#pragma once
#include "optimization/optimizer.h"
#include "util/host_param_view.h"
#include "util/cuda_device_param_view.h"
#include "util/device_platform_handler_cuda.h"
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cmath>
#include <CL/cl.h>

typedef struct CUstream_st* cudaStream_t;

/**
 * @brief Implements the Adam optimizer using CUDA device execution.
 */
class AdamOptimizerCu : public Optimizer
{
public:
    /**
     * @brief Constructs the CUDA Adam optimizer with explicit hyperparameters.
     * @param lr Learning rate.
     * @param beta1 Exponential decay factor for the first moment estimate.
     * @param beta2 Exponential decay factor for the second moment estimate.
     * @param eps Numerical stability epsilon.
     */
    AdamOptimizerCu(float lr, float beta1, float beta2, float eps)
        : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {}

    /**
     * @brief Performs one optimization step for all provided parameters.
     * @param benchmarkData Benchmark row updated with execution metadata.
     * @param params Host-side parameter views to optimize.
     * @param step_index Iteration index of the optimization step.
     */
    void step(BenchmarkData *benchmarkData, const std::vector<HostParamView> &params, int step_index);

    /**
     * @brief Creates a CUDA parameter view for a host parameter tensor.
     * @param p Host-side parameter view.
     * @return Pointer to the created CUDA device parameter view.
     */
    CudaDeviceParamView *convertHostToDevice(const HostParamView &p);

    /**
     * @brief Launches the CUDA Adam update kernel for one parameter tensor.
     * @param params CUDA device parameter view.
     * @param d_m Device buffer storing the first moment estimate.
     * @param d_v Device buffer storing the second moment estimate.
     * @param lr Learning rate.
     * @param beta1 Exponential decay factor for the first moment estimate.
     * @param beta2 Exponential decay factor for the second moment estimate.
     * @param bc1 Bias-correction term for the first moment estimate.
     * @param bc2 Bias-correction term for the second moment estimate.
     * @param eps Numerical stability epsilon.
     * @param stream CUDA stream used for kernel execution.
     */
    void launch_adam_update(
        CudaDeviceParamView* params,
        float *d_m,
        float *d_v,
        float lr,
        float beta1,
        float beta2,
        float bc1,
        float bc2,
        float eps,
        cudaStream_t stream);

    /**
     * @brief Stores cached CUDA buffers for one parameter tensor.
     */
    struct State {
        size_t n = 0;
        float* m = nullptr;
        float* v = nullptr;

        float* d_param = nullptr; // staging buffer
        float* d_grad  = nullptr; // staging buffer
    };

private:
    float lr_, beta1_, beta2_, eps_;
    std::unordered_map<const float *, State> states_;

    /**
     * @brief Returns the cached CUDA state for a parameter buffer.
     * @param hp Host parameter view used as the cache key.
     * @return Reference to the cached CUDA state.
     */
    State &state_for_(const HostParamView &hp);
    Logger logger;
};
