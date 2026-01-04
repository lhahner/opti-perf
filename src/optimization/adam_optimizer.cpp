#include "optimization/adam_optimizer.h"

void AdamOptimizer::step(const std::vector<HostParamView>& params, int step_index)  {
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

AdamOptimizer::State& AdamOptimizer::state_for_(const HostParamView& p)
{
	auto it = states_.find(p.data);
        if (it == states_.end()) {
            State st;
            st.m.assign(p.count, 0.0f);
            st.v.assign(p.count, 0.0f);
            it = states_.emplace(p.data, std::move(st)).first;
        } else {
            if (it->second.m.size() != p.count) {
                it->second.m.assign(p.count, 0.0f);
                it->second.v.assign(p.count, 0.0f);
            }
        }
        return it->second;
}

