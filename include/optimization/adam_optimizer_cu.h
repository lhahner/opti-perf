// optimization/adam_optimizer.h
#pragma once
#include "optimization/optimizer.h"
#include "util/host_param_view.h"
#include "util/cuda_device_param_view.h"
#include <unordered_map>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include <CL/cl.h>

class AdamOptimizer : public Optimizer
{
public:
    AdamOptimizer(float lr, float beta1, float beta2, float eps)
        : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {}

    void step(const std::vector<HostParamView> &params, int step_index);

    void adam_update_kernel(
        float *__restrict__ param,      // p.data (device)
        const float *__restrict__ grad, // p.grad (device)
        float *__restrict__ m,          // st.m (device)
        float *__restrict__ v,          // st.v (device)
        size_t n,
        float lr,
        float beta1,
        float beta2,
        float bc1, // (1 - beta1^t)
        float bc2, // (1 - beta2^t)
        float eps);

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

    class State
    {
    public:
        float *m = nullptr; // device
        float *v = nullptr; // device
        size_t n = 0;       // number of elements currently allocated
    };

private:
    float lr_, beta1_, beta2_, eps_;
    std::unordered_map<const float *, State> states_;
    State& state_for_(const CudaDeviceParamView* p);
};
