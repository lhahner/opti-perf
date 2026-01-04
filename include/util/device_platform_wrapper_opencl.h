#include "CL/opencl.h"
#include <CL/cl.h>
#include <stdexcept>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "util/setup_wrapper.h"

class DevicePlatformWrapperOpenCL {
	public:
		int setup();
		static DevicePlatformWrapperOpenCL* getInstance();
		
		cl_context createContext();
		cl_context getClContext();
		void setClContext(cl_context);

		cl_command_queue createCommandQueue(cl_context context, 
				cl_device_id *device);
		cl_command_queue getClCommandQueueForDevice();

		cl_program createProgram(cl_context context, cl_device_id device,
				const char* kernel);

		cl_device_id getDeviceId();
	private:
		static DevicePlatformWrapperOpenCL* devicePlatformWrapperOpenCL_;
		const char* kernel_name_;
		const char* kernelFile_;
		size_t deviceBufferSize = -1;
		cl_context context = 0;
		cl_command_queue commandQueue = 0;
		cl_program program = 0;
		cl_device_id device = 0;
		cl_kernel kernel = 0;
		cl_mem memObjects[N_MEMOBJ] = {};
		cl_int errNum;
};
