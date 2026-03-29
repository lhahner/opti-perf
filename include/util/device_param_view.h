// adam_optimizer_cl.h
#pragma once

#include <CL/opencl.h>

/**
 * @brief Holds OpenCL device buffers for a single optimizer parameter tensor.
 */
class DeviceParamView {
	public:
		cl_mem param;     
		cl_mem grad;
		cl_mem m;    
		cl_mem v;    
		int n;
};
