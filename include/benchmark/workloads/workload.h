#ifndef INCLUDE_BENCHMARK_WORKLOADS_WORKLOAD_H_
#define INCLUDE_BENCHMARK_WORKLOADS_WORKLOAD_H_

#include <map>
#include <vector>
#include "util/host_param_view.h"

class Workload {
	public:
		virtual ~Workload() = default;
		virtual void runForward() = 0;
		virtual std::pair<int, float> computeLoss() = 0;
		virtual std::vector<HostParamView> parameters() = 0;
};

#endif
