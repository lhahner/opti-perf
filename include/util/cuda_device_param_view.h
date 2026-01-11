#pragma once

class CudaDeviceParamView 
{
    public:
        float* param;
        float* grad;
        float* m;
        float* v;
        int n;
};