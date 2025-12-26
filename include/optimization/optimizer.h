#ifndef INCLUDE_OPTIMIZATION_OPTIMIZER_H_
#define INCLUDE_OPTIMIZATION_OPTIMIZER_H_

#pragma once
#include <vector>
#include <cstddef>
#include "util/host_param_view.h"

class Optimizer {
	public:
		virtual ~Optimizer() = default;
		virtual void step(const std::vector<HostParamView>& params, int step_index) = 0;
};

#endif /* INCLUDE_OPTIMIZATION_OPTIMIZER_H_ */
