// adam_optimizer_cl.h
#pragma once

#include <CL/opencl.h>

class DeviceParamView {
	public:
		cl_mem param;     
		cl_mem grad;
		cl_mem m;    
		cl_mem v;    
		int n;
};
