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

/**
 * @brief Implements a dense matrix multiplication workload with least-squares loss.
 */
class GEMM : public Workload {
public:
    /**
     * @brief Constructs the workload from matrix dimensions.
     * @param dimensions Dimension triple `{M, K, N}`.
     */
	explicit GEMM(const std::vector<int>& dimensions);

    /**
     * @brief Constructs the workload from an initializer list of dimensions.
     * @param dims Dimension triple `{M, K, N}`.
     */
	explicit GEMM(std::initializer_list<int> dims)
	        : GEMM(std::vector<int>(dims)) {}

    /**
     * @brief Allocates and initializes workload input, target, and parameter buffers.
     */
    void initializeInput();

    /**
     * @brief Executes the forward pass and computes gradients for the workload.
     */
    void runForward();

    /**
     * @brief Returns the current training step and loss value.
     * @return Pair containing the step index and current loss.
     */
    std::pair<int, float> computeLoss();

    /**
     * @brief Returns the host-side parameter views exposed to the optimizer.
     * @return Collection of parameter and gradient views.
     */
    std::vector<HostParamView> parameters();

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
    /**
     * @brief Computes a row-major matrix multiplication.
     * @param M Number of rows in `A` and `C`.
     * @param N Number of columns in `B` and `C`.
     * @param K Shared inner dimension of `A` and `B`.
     * @param A Left input matrix in row-major layout.
     * @param B Right input matrix in row-major layout.
     * @param C Output matrix in row-major layout.
     */
    static void gemm_rowmajor(
        int M, int N, int K,
        const float* A, // MxK
        const float* B, // KxN
        float* C        // MxN
    );

    /**
     * @brief Transposes a row-major matrix into another row-major buffer.
     * @param rows Number of source rows.
     * @param cols Number of source columns.
     * @param src Source matrix buffer.
     * @param dst Destination buffer receiving the transposed matrix.
     */
    static void transpose_rowmajor(
        int rows, int cols,
        const float* src, // rows x cols
        float* dst         // cols x rows
    );

    /**
     * @brief Computes the mean squared error term used by the workload loss.
     * @param E Residual buffer.
     * @param count Number of residual values.
     * @return Mean squared error value.
     */
    static float mse_half(const float* E, size_t count);
};

#endif
