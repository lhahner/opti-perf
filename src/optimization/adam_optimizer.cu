#include <cuda_runtime.h>
#include <iostream>
#include "optimization/adam_optimizer_cu.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

namespace {
constexpr int kWarmupSteps = 10;

const char *format_timestamp()
{
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&tt, &tm);
    std::ostringstream ts;
    ts << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
    static thread_local std::string ts_str;
    ts_str = ts.str();
    return ts_str.c_str();
}
} // namespace

struct CudaEventTimer {
    cudaEvent_t start, stop;

    CudaEventTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~CudaEventTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void begin(cudaStream_t stream) {
        cudaEventRecord(start, stream);
    }

    float end(cudaStream_t stream) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

void AdamOptimizerCu::step(BenchmarkData *benchmarkData, const std::vector<HostParamView> &params, int step_index)
{
    const int t = (step_index < 1) ? 1 : step_index;

    const float b1t = std::pow(beta1_, (float)t);
    const float b2t = std::pow(beta2_, (float)t);
    const float bc1 = 1.0f - b1t;
    const float bc2 = 1.0f - b2t;

    cudaStream_t stream = 0;

    float h2d_ms_total = 0.0f;
    float kernel_ms_total = 0.0f;
    float d2h_ms_total = 0.0f;
    DevicePlatformHandlerCuda device_platform_handler_cuda;
    if (benchmarkData != nullptr)
    {
        benchmarkData->device_name = device_platform_handler_cuda.get_device_name();
    }
    for (const auto &p : params)
    {
        if (!p.data || !p.grad || p.count == 0)
            continue;

        State &st = state_for_(p);

        {
            CudaEventTimer timer;
            timer.begin(stream);

            cudaMemcpyAsync(st.d_param, p.data, p.count * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(st.d_grad, p.grad, p.count * sizeof(float),
                            cudaMemcpyHostToDevice, stream);

            h2d_ms_total += timer.end(stream);
        }

        {
            CudaDeviceParamView dv;
            dv.param = st.d_param;
            dv.grad  = st.d_grad;
            dv.n     = p.count;

            CudaEventTimer timer;
            timer.begin(stream);

            launch_adam_update(&dv, st.m, st.v,
                               lr_, beta1_, beta2_, bc1, bc2, eps_,
                               stream);

            kernel_ms_total += timer.end(stream);
        }

        {
            CudaEventTimer timer;
            timer.begin(stream);

            cudaMemcpyAsync(p.data, st.d_param, p.count * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);

            d2h_ms_total += timer.end(stream);
        }
    }
    cudaStreamSynchronize(stream);

    if (benchmarkData != nullptr && step_index > kWarmupSteps)
    {
        benchmarkData->timestamp = format_timestamp();
        benchmarkData->workload_type = "h2d_transfer";
        benchmarkData->time_ms = h2d_ms_total;
        logger.logToCsv(*benchmarkData, benchmarkData->log_filename.c_str());

        benchmarkData->timestamp = format_timestamp();
        benchmarkData->workload_type = "compute";
        benchmarkData->time_ms = kernel_ms_total;
        logger.logToCsv(*benchmarkData, benchmarkData->log_filename.c_str());

        benchmarkData->timestamp = format_timestamp();
        benchmarkData->workload_type = "d2h_transfer";
        benchmarkData->time_ms = d2h_ms_total;
        logger.logToCsv(*benchmarkData, benchmarkData->log_filename.c_str());
    }

    std::cout << toString(Marker::INFO) << "Data Transfer: " << d2h_ms_total
              << " ms ,Execution Time: " << kernel_ms_total
              << " ms ,Total: " << (h2d_ms_total + kernel_ms_total + d2h_ms_total)
              << " ms\n";
}

namespace adam_kernels
{

    __global__ void adam_update_kernel_impl(
        float *__restrict__ param,
        const float *__restrict__ grad,
        float *__restrict__ m,
        float *__restrict__ v,
        size_t n,
        float lr, float beta1, float beta2,
        float bc1, float bc2, float eps)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
            return;

        float g = grad[i];
        float mi = beta1 * m[i] + (1.0f - beta1) * g;
        float vi = beta2 * v[i] + (1.0f - beta2) * (g * g);

        m[i] = mi;
        v[i] = vi;

        float mhat = mi / bc1;
        float vhat = vi / bc2;

        param[i] -= lr * mhat / (sqrtf(vhat) + eps);
    }
} 

void AdamOptimizerCu::launch_adam_update(
    CudaDeviceParamView *deviceParamView,
    float *d_m,
    float *d_v,
    float lr,
    float beta1,
    float beta2,
    float bc1,
    float bc2,
    float eps,
    cudaStream_t stream)
{
    const int threads = 256;
    const int blocks = (int)((deviceParamView->n + threads - 1) / threads);

    adam_kernels::adam_update_kernel_impl<<<blocks, threads, 0, stream>>>(
        deviceParamView->param,
        deviceParamView->grad,
        d_m, d_v,
        deviceParamView->n,
        lr, beta1, beta2,
        bc1, bc2, eps);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cerr << toString(Marker::ERROR) << "adam_update kernel launch failed: " << cudaGetErrorString(e) << "\n";
        std::abort();
    }
}

CudaDeviceParamView *AdamOptimizerCu::convertHostToDevice(const HostParamView &p)
{
    CudaDeviceParamView *deviceParamView = new CudaDeviceParamView();
    deviceParamView->grad = p.grad;
    deviceParamView->param = p.data;
    deviceParamView->n = p.count;
    return deviceParamView;
}

AdamOptimizerCu::State &AdamOptimizerCu::state_for_(const HostParamView &hp)
{
    auto it = states_.find(hp.data); 
    if (it == states_.end()) {
        State st{};
        st.n = hp.count;

        cudaMalloc(&st.m, st.n * sizeof(float));
        cudaMalloc(&st.v, st.n * sizeof(float));
        cudaMemset(st.m, 0, st.n * sizeof(float));
        cudaMemset(st.v, 0, st.n * sizeof(float));

        cudaMalloc(&st.d_param, st.n * sizeof(float));
        cudaMalloc(&st.d_grad,  st.n * sizeof(float));

        it = states_.emplace(hp.data, st).first;
    } else {
        if (it->second.n != hp.count) {
            State& st = it->second;
            cudaFree(st.m); cudaFree(st.v);
            cudaFree(st.d_param); cudaFree(st.d_grad);

            st.n = hp.count;
            cudaMalloc(&st.m, st.n * sizeof(float));
            cudaMalloc(&st.v, st.n * sizeof(float));
            cudaMemset(st.m, 0, st.n * sizeof(float));
            cudaMemset(st.v, 0, st.n * sizeof(float));

            cudaMalloc(&st.d_param, st.n * sizeof(float));
            cudaMalloc(&st.d_grad,  st.n * sizeof(float));
        }
    }
    return it->second;
}
