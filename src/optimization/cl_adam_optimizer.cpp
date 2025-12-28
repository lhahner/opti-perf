#include "optimization/cl_adam_optimizer.h"

ClAdamOptimizer::ClAdamOptimizer(float lr, float beta1, float beta2,
		float eps) :
		lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
}

State& ClAdamOptimizer::state_for_(const HostParamView &p) {
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
