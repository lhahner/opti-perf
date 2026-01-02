// optimization/adam_optimizer.h
#pragma once
#include "optimization/optimizer.h"
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cmath>
#include <CL/cl.h>

class AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(float lr, float beta1, float beta2, float eps)
        : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {}

    void step(const std::vector<HostParamView>& params, int step_index) override {
        // Precompute bias corrections once per global step
        const float b1t = std::pow(beta1_, (float)step_index);
        const float b2t = std::pow(beta2_, (float)step_index);
        const float bc1 = 1.0f - b1t;
        const float bc2 = 1.0f - b2t;

        for (const auto& p : params) {
            if (!p.data || !p.grad || p.count == 0) continue;

            State& st = state_for_(p);

            // Adam update
            for (size_t i = 0; i < p.count; ++i) {
                const float g = p.grad[i];
                st.m[i] = beta1_ * st.m[i] + (1.0f - beta1_) * g;
                st.v[i] = beta2_ * st.v[i] + (1.0f - beta2_) * (g * g);

                const float mhat = st.m[i] / bc1;
                const float vhat = st.v[i] / bc2;

                p.data[i] -= lr_ * mhat / (std::sqrt(vhat) + eps_);
            }
        }
    }

private:
    struct State {
        std::vector<float> m;
        std::vector<float> v;
    };

    float lr_, beta1_, beta2_, eps_;

    // Key by the parameter pointer identity (valid as long as buffer storage doesn't reallocate).
    std::unordered_map<const float*, State> states_;

    State& state_for_(const HostParamView& p) {
        auto it = states_.find(p.data);
        if (it == states_.end()) {
            State st;
            st.m.assign(p.count, 0.0f);
            st.v.assign(p.count, 0.0f);
            it = states_.emplace(p.data, std::move(st)).first;
        } else {
            // If shape changes (shouldn't in benchmarks), resize.
            if (it->second.m.size() != p.count) {
                it->second.m.assign(p.count, 0.0f);
                it->second.v.assign(p.count, 0.0f);
            }
        }
        return it->second;
    }
};
