#ifndef INCLUDE_BENCHMARK_WORKLOADS_GENERALMATRIXMULTIPLICATION_GEMM_H_
#define INCLUDE_BENCHMARK_WORKLOADS_GENERALMATRIXMULTIPLICATION_GEMM_H_

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <iostream>

#include "benchmark/workloads/workload.h"

class GEMM : public Workload {
public:
    // dimensions = {M, K, N}
	explicit GEMM(const std::vector<int>& dimensions);
	explicit GEMM(std::initializer_list<int> dims)
	        : GEMM(std::vector<int>(dims)) {}

    void initializeInput();   // allocate + initialize W, X, Y*
    void runForward();        // compute Y, residual, dW, loss
    std::pair<int, float> computeLoss(); // (step, loss)
    std::vector<HostParamView> parameters(); // expose W + gradW

    const char* workloadType = "GEMM";
    const char* workloadName = "LeastSquares (MLP-like)";

    const std::vector<int> dimensions_;

private:
    // Shapes
    int M_{0}, K_{0}, N_{0};

    // Training step counter
    int step_{0};
    float loss_{0.0f};

    // Buffers (row-major)
    // W: MxK (parameters)
    // X: KxN (inputs)
    // Yt: MxN (targets)
    // Y: MxN (predictions)
    // E: MxN (residual)
    // dW: MxK (gradients)

    std::vector<float> W_;
    std::vector<float> X_;
    std::vector<float> Yt_;
    std::vector<float> Y_;
    std::vector<float> E_;
    std::vector<float> dW_;

    // Scratch for transpose of X (Xt: NxK) if you want explicit transpose
    // Optional: you can implement GEMM that reads X as transposed without materializing.
    std::vector<float> Xt_;

private:
    static void gemm_rowmajor(
        int M, int N, int K,
        const float* A, // MxK
        const float* B, // KxN
        float* C        // MxN
    );

    static void transpose_rowmajor(
        int rows, int cols,
        const float* src, // rows x cols
        float* dst         // cols x rows
    );

    static float mse_half(const float* E, size_t count);
};

#endif
