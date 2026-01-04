#pragma once

#include "optimization/optimizer.h"
#include <CL/cl.h> 
#include <unordered_map> 
#include <vector> 
#include <cmath> 
#include <stdexcept> 
#include "CL/opencl.h" 
#include "util/device_param_view.h" 
#include "util/host_param_view.h" 

class AdamOptimizerCl : public Optimizer { 
	public: 
		AdamOptimizerCl() = default;
		void step(const std::vector<HostParamView>& params, int step_index);	
		void step_one_tensor(cl_command_queue queue, cl_kernel adam_kernel, DeviceParamView& dv,
				int step_index, float lr, float beta1, float beta2, float eps,
				size_t local_size);
		DeviceParamView& toDevice(cl_context context, 
				cl_command_queue queue, 
				const HostParamView& parameters); 
		void fromDevice(cl_command_queue q, 
				DeviceParamView& dv, 
				const HostParamView& hp);

	private: 
		std::unordered_map<const float*, DeviceParamView> device_state_; 
		cl_context context_ = nullptr;
		cl_command_queue queue_ = nullptr;
		cl_kernel kernel_ = nullptr;
		float lr_ = 1e-3f, beta1_ = 0.9f, beta2_ = 0.999f, eps_ = 1e-8f;	
		size_t local_size_ = 256;

	public:
		void configure(cl_context ctx, cl_command_queue q, cl_kernel k,
				float lr, float b1, float b2, float eps, size_t local_size); 
};
