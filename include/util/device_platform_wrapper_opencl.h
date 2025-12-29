#include "CL/opencl.h"
#include <stdexcept>
#include <iostream>

// Memory object size
#define N_MEMOBJ 3

// General Setup flags
#define SETUP_SUCCESS 0
#define SETUP_FAILURE 1

// Context creation flags
#define CONTEXT_SUCCESS 0
#define CONTEXT_FAILURE 1

class DevicePlatformWrapperOpenCL {
public:
	void queueKernel();
	void readBuffer();
	int initSetup();
	
	cl_context createContext();
	cl_context getClContext();
	void setClContext(cl_context);

	cl_command_queue getClCommandQueueForDevice();
	void setClCommandQueueForDevice(cl_command_queue commandQueue);

	cl_program getClProgram();
	void setClProgram(cl_program program);
	
	cl_kernel getClKernel();
	void setClKernel(cl_kernel kernel);	
	
	cl_mem getClMemory();
	void setClMemory(cl_mem memObjects[N_MEMOBJ]);

private:
	cl_context context = 0;
	cl_command_queue commandQueue = 0;
	cl_program program = 0;
	cl_device_id device = 0;
	cl_kernel kernel = 0;
	cl_mem memObjects[N_MEMOBJ] = {};
	cl_int errNum;
};
