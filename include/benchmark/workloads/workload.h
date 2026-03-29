#ifndef INCLUDE_BENCHMARK_WORKLOADS_WORKLOAD_H_
#define INCLUDE_BENCHMARK_WORKLOADS_WORKLOAD_H_

#include <map>
#include <vector>
#include "util/host_param_view.h"

/**
 * @brief Defines the common interface for benchmark workloads.
 */
class Workload {
	public:
		/**
		 * @brief Destroys the workload instance.
		 */
		virtual ~Workload() = default;

		/**
		 * @brief Executes the forward pass of the workload.
		 */
		virtual void runForward() = 0;

		/**
		 * @brief Computes the current workload loss value.
		 * @return Pair containing workload-specific loss metadata and the loss value.
		 */
		virtual std::pair<int, float> computeLoss() = 0;

		/**
		 * @brief Returns the parameter views optimized by the workload.
		 * @return Collection of host-side parameter views.
		 */
		virtual std::vector<HostParamView> parameters() = 0;
};

#endif
