#pragma once

/**
 * @brief Holds CUDA device pointers for a single optimizer parameter tensor.
 */
class CudaDeviceParamView 
{
    public:
        float* param;
        float* grad;
        float* m;
        float* v;
        int n;
};
