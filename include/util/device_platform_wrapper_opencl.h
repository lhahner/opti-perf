#include "CL/opencl.h"
#include <CL/cl.h>
#include <stdexcept>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "util/setup_wrapper.h"

class DevicePlatformWrapperOpenCL : public SetupWrapper {
	public:
		void queueKernel();
		void readBuffer();
		int setup();
		SetupWrapper* getInstance(const char* kernel_name, const char* kernelFile);
		
		cl_context createContext();
		cl_context getClContext();
		void setClContext(cl_context);

		cl_command_queue createCommandQueue(cl_context context, 
				cl_device_id *device);
		cl_command_queue getClCommandQueueForDevice();
		void setClCommandQueueForDevice(cl_command_queue commandQueue);

		cl_program createProgram(cl_context context, cl_device_id device,
				const char* kernel);
		cl_program getClProgram();
		void setClProgram(cl_program program);

		cl_kernel getClKernel();
		void setClKernel(cl_kernel kernel);	

		cl_mem createMemoryObjects(cl_context context, cl_mem* memObjects);	
		cl_mem getClMemory();
		void setClMemory(cl_mem memObjects[N_MEMOBJ]);
		int cleanup(cl_context context, 
				cl_command_queue commandQueue,
				cl_program program, 
				cl_kernel kernel, 
				cl_mem* memObjects);

	protected:
		DevicePlatformWrapperOpenCL(const char* kernel_name, const char* kernel_file)
		: kernel_name_(kernel_name), kernelFile_(kernel_file) {}		
	
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
