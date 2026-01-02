#include <CL/opencl.h>

class DeviceTensorAdam {
	public:
		cl_mem param;     
		cl_mem grad;
		cl_mem m;    
		cl_mem v;    
		int n;
};
