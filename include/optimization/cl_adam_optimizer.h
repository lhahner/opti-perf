#ifndef INCLUDE_OPTIMIZATION_ADAM_OPTIMIZER_OPENCL_CL_
#define INCLUDE_OPTIMIZATION_ADAM_OPTIMIZER_OPENCL_CL_

#include "optimization/optimizer.h"
#include <CL/opencl.hpp>

struct State {
        std::vector<float> m;
        std::vector<float> v;
};

__kernel void precomputeBiasCorrections(float& b1t, float& b2t, float& bc1);
__kernel void update(const std::vector<HostParamView>& params, float beta1_, float beta2_, float bc1, float bc2, float eps, State& state);

class ClAdamOptimizer : public Optimizer {
public:
	ClAdamOptimizer(float lr, float beta1, float beta2, float eps);

	float getBeta1() const {
		return beta1_;
	}

	void setBeta1(float beta1) {
		beta1_ = beta1;
	}

	float getBeta2() const {
		return beta2_;
	}

	void setBeta2(float beta2) {
		beta2_ = beta2;
	}

	float getEps() const {
		return eps_;
	}

	void setEps(float eps) {
		eps_ = eps;
	}

	float getLr() const {
		return lr_;
	}

	void setLr(float lr) {
		lr_ = lr;
	}

	const std::unordered_map<const float*, State>& getStates() const {
		return states_;
	}

	void setStates(const std::unordered_map<const float*, State> &states) {
		states_ = states;
	}

private:
	 float lr_, beta1_, beta2_, eps_;
	 std::unordered_map<const float*, State> states_;

	 State& state_for_(const HostParamView& p);
};

#endif
