__kernel void precomputeBiasCorrections(float b1t, float b2t, float bc1, float bc2, int step_index, ClAdamOptimizer clAdamOptimizer) {
	b1t = pow(clAdamOptimizer->getBeta1(), (float) step_index);
	b2t = pow(clAdamOptimizer->getBeta2(), (float) step_index);
	bc1 = 1.0f - b1t;
	bc2 = 1.0f - b2t;
	return;
}

__kernel void update(const std::vector<HostParamView> &params, float beta1_,
		float beta2_, float bc1, float bc2, float eps, State &state) {
	for (const auto &p : params) {
		if (!p.data || !p.grad || p.count == 0)
			continue;

		State &st = state_for_(p);

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
