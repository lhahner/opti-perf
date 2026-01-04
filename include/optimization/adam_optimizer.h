// optimization/adam_optimizer.h
#pragma once
#include "optimization/optimizer.h"
#include "util/host_param_view.h"
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cmath>
#include <CL/cl.h>

class AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(float lr, float beta1, float beta2, float eps)
        : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {}
    void step(const std::vector<HostParamView>& params, int step_index);

private:
   struct State {
        std::vector<float> m;
        std::vector<float> v;
    }; 
    float lr_, beta1_, beta2_, eps_;
    std::unordered_map<const float*, State> states_;
    State& state_for_(const HostParamView& p);
};
