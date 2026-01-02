
#include "optimization/optimizer.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include "CL/opencl.h"
#include "util/device_tensor_adam.h"
#include "util/host_param_view.h"

class AdamOptimizerCl {
	public:
		AdamOptimizerCl() = default;
		void step(cl_command_queue queue,
			  cl_kernel adam_kernel,   
			  const std::vector<DeviceTensorAdam>& tensors,
			  int step_index, 
			  float lr, float beta1, float beta2, float eps,
			  size_t local_size);
		std::vector<DeviceTensorAdam>* transferToBuffer(std::vector<HostParamView> parameters); 
};
