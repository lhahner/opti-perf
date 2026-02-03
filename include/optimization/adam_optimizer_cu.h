// optimization/adam_optimizer.h
#pragma once
#include "optimization/optimizer.h"
#include "util/host_param_view.h"
#include "util/cuda_device_param_view.h"
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cmath>
#include <CL/cl.h>

typedef struct CUstream_st* cudaStream_t;

class AdamOptimizerCu : public Optimizer
{
public:
    AdamOptimizerCu(float lr, float beta1, float beta2, float eps)
        : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {}

    void step(BenchmarkData *benchmarkData, const std::vector<HostParamView> &params, int step_index);

    CudaDeviceParamView *convertHostToDevice(const HostParamView &p);

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
    State &state_for_(const HostParamView &hp);
    Logger logger;
};
