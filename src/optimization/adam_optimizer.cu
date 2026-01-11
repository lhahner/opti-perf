#include "optimization/adam_optimizer_cu.h"

void AdamOptimizer::step(const std::vector<HostParamView>& params, int step_index)  {
	 const int t = step_index + 1;

    const float b1t = std::pow(beta1_, (float)t);
    const float b2t = std::pow(beta2_, (float)t);
    const float bc1 = 1.0f - b1t;
    const float bc2 = 1.0f - b2t;

    cudaStream_t stream = 0; // default stream

    for (const auto& p : params) {
        if (!p.data || !p.grad || p.count == 0) continue;
        CudaDeviceParamView* deviceParamView = convertHostToDevice(p);
        State& st = state_for_(deviceParamView);

        launch_adam_update(
            deviceParamView, st.m, st.v, 
            lr_, beta1_, beta2_, bc1, bc2, eps_,
            stream
        );
    }
}

void AdamOptimizer::launch_adam_update(
    CudaDeviceParamView* deviceParamView,
    float *d_m,
    float *d_v,
    float lr,
    float beta1,
    float beta2,
    float bc1,
    float bc2,
    float eps,
    cudaStream_t stream = 0
) {

    const int threads = 256;
    const int blocks  = (int)((deviceParamView->n + threads - 1) / threads);

    adam_update_kernel<<<blocks, threads, 0, stream>>>(
        deviceParamView->param, deviceParamView->grad, d_m, d_v, deviceParamView->n, lr, beta1, beta2, bc1, bc2, eps
    );
}

CudaDeviceParamView* AdamOptimizer::convertHostToDevice(const HostParamView &p){
    CudaDeviceParamView *deviceParamView = new CudaDeviceParamView();
    deviceParamView->grad = p.grad;
    deviceParamView->param = p.data;
    deviceParamView->n = p.count;
    return deviceParamView;
}

__global__ void AdamOptimizer::adam_update_kernel(
    float* __restrict__ param,      // p.data (device)
    const float* __restrict__ grad,  // p.grad (device)
    float* __restrict__ m,           // st.m (device)
    float* __restrict__ v,           // st.v (device)
    size_t n,
    float lr,
    float beta1,
    float beta2,
    float bc1,   // (1 - beta1^t)
    float bc2,   // (1 - beta2^t)
    float eps
) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float g  = grad[i];

    const float mi = beta1 * m[i] + (1.0f - beta1) * g;
    const float vi = beta2 * v[i] + (1.0f - beta2) * (g * g);

    m[i] = mi;
    v[i] = vi;

    const float mhat = mi / bc1;
    const float vhat = vi / bc2;

    param[i] -= lr * mhat / (sqrtf(vhat) + eps);
}

AdamOptimizer::State& AdamOptimizer::state_for_(const CudaDeviceParamView* p)
{
    // Keyed by the parameter device pointer
    auto it = states_.find(p->param);
    if (it == states_.end()) {
        // Default-constructed State has nullptrs and n=0
        it = states_.emplace(p->param, State{}).first;
    }
    return it->second;
}