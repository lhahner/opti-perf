#include "benchmark/workloads/generalmatrixmultiplication/gemm.h"

#include <algorithm>
#include <cmath>
#include <random>

GEMM::GEMM(const std::vector<int>& dimensions)
    : dimensions_(dimensions)
{
    if (dimensions_.size() != 3) {
        throw std::invalid_argument("GEMM dimensions must be {M,K,N}");
    }
    M_ = dimensions_[0];
    K_ = dimensions_[1];
    N_ = dimensions_[2];
    if (M_ <= 0 || K_ <= 0 || N_ <= 0) {
        throw std::invalid_argument("M,K,N must be positive");
    }
}

void GEMM::initializeInput() {
    // Allocate
    W_.assign((size_t)M_ * (size_t)K_, 0.0f);
    X_.assign((size_t)K_ * (size_t)N_, 0.0f);
    Yt_.assign((size_t)M_ * (size_t)N_, 0.0f);

    Y_.assign((size_t)M_ * (size_t)N_, 0.0f);
    E_.assign((size_t)M_ * (size_t)N_, 0.0f);
    dW_.assign((size_t)M_ * (size_t)K_, 0.0f);

    Xt_.assign((size_t)N_ * (size_t)K_, 0.0f);

    // Initialize deterministically for reproducible benchmarks
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    for (auto& v : W_) v = dist(rng);
    for (auto& v : X_) v = dist(rng);

    // Create a fixed target using a "true" matrix W_true so the problem is solvable
    std::vector<float> Wtrue((size_t)M_ * (size_t)K_, 0.0f);
    for (auto& v : Wtrue) v = dist(rng);

    // Yt = Wtrue * X
    gemm_rowmajor(M_, N_, K_, Wtrue.data(), X_.data(), Yt_.data());

    step_ = 0;
    loss_ = 0.0f;
}

void GEMM::runForward() {
    if (W_.empty()) {
        throw std::logic_error("Call initializeInput() before runForward()");
    }

    // 1) Forward: Y = W * X
    gemm_rowmajor(M_, N_, K_, W_.data(), X_.data(), Y_.data());

    // 2) Residual: E = Y - Yt
    const size_t MN = (size_t)M_ * (size_t)N_;
    for (size_t i = 0; i < MN; ++i) {
        E_[i] = Y_[i] - Yt_[i];
    }

    // 3) Loss: 0.5 * mean(E^2)
    // mean over M*N so loss scale stays consistent across sizes
    loss_ = 0.5f * mse_half(E_.data(), MN);

    // 4) Gradient: dW = (1/(M*N)) * E * X^T
    // Compute Xt (transpose X: KxN -> NxK)
    transpose_rowmajor(K_, N_, X_.data(), Xt_.data()); // Xt is N x K

    // dW = E(MxN) * Xt(NxK) => MxK
    gemm_rowmajor(M_, K_, N_, E_.data(), Xt_.data(), dW_.data());

    // Scale gradient by 1/(M*N)
    const float scale = 1.0f / static_cast<float>(MN);
    const size_t MK = (size_t)M_ * (size_t)K_;
    for (size_t i = 0; i < MK; ++i) {
        dW_[i] *= scale;
    }

    ++step_;
}

std::pair<int, float> GEMM::computeLoss() {
    return {step_, loss_};
}

std::vector<HostParamView> GEMM::parameters() {
    // Expose W and dW to an external optimizer
    HostParamView pv;
    pv.data = W_.data();
    pv.grad = dW_.data();
    pv.count = W_.size();
    pv.name = "W";
    return {pv};
}

// ---------------- Helpers ----------------

void GEMM::gemm_rowmajor(int M, int N, int K,
                         const float* A, const float* B, float* C)
{
    // Naive GEMM, row-major:
    // C[m,n] = sum_k A[m,k] * B[k,n]
    // Initialize C to 0 each time (explicit).
    std::fill(C, C + (size_t)M * (size_t)N, 0.0f);

    for (int m = 0; m < M; ++m) {
        const float* Arow = A + (size_t)m * (size_t)K;
        float* Crow = C + (size_t)m * (size_t)N;
        for (int k = 0; k < K; ++k) {
            const float a = Arow[k];
            const float* Brow = B + (size_t)k * (size_t)N;
            for (int n = 0; n < N; ++n) {
                Crow[n] += a * Brow[n];
            }
        }
    }
}

void GEMM::transpose_rowmajor(int rows, int cols,
                              const float* src, float* dst)
{
    // src: rows x cols, dst: cols x rows
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            dst[(size_t)c * (size_t)rows + (size_t)r] = src[(size_t)r * (size_t)cols + (size_t)c];
        }
    }
}

float GEMM::mse_half(const float* E, size_t count) {
    double acc = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double e = (double)E[i];
        acc += e * e;
    }
    return (float)(acc / (double)count);
}
