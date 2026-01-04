#include "util/device_platform_wrapper_opencl.h"
#include <CL/cl.h>
#include <stdexcept>

DevicePlatformWrapperOpenCL* DevicePlatformWrapperOpenCL::devicePlatformWrapperOpenCL_  = nullptr;;

/**
 * Core Setup and Implementation
 *
 * @return 0 if success and 1 if not, logs failure cause anyway.
 **/
int DevicePlatformWrapperOpenCL::setup() 
{
	this->context = this->createContext();
	if (this->getClContext() == NULL) {
		std::cerr 
			<< "Context Creation in inital setup for OpenCL failed." 
			<< std::endl;
		return SETUP_FAILURE;	
	}

	this->commandQueue = this->createCommandQueue(
			this->context, 
			&this->device);

	if (this->commandQueue == NULL) 
	{
		std::cerr << "Failed to created commandQueue in inital seutp."
			<< std::endl;
		return SETUP_FAILURE;
	}
	this->program = this->createProgram(this->context, this->device, this->kernelFile_);
	if (this->program == NULL) 
	{	
		std::cerr
			<< "Program creation in inital setup for OpenCL failed."
			<< std::endl;
		return SETUP_FAILURE;
	}
	return SETUP_SUCCESS;
}

DevicePlatformWrapperOpenCL* DevicePlatformWrapperOpenCL::getInstance()
{
	if (devicePlatformWrapperOpenCL_ == nullptr) 
	{
		devicePlatformWrapperOpenCL_ = 
			new DevicePlatformWrapperOpenCL();
		return devicePlatformWrapperOpenCL_;
	}
	return devicePlatformWrapperOpenCL_;
}


/**
 * Create an OpenCL context on the first available platform.
 *
 * @return returns the created context with the first available paltform.  
 **/
cl_context DevicePlatformWrapperOpenCL::createContext()
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId;
	cl_context context = NULL;
	errNum = clGetPlatformIDs(1, 
			&firstPlatformId, 
			&numPlatforms);

	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		std::cerr << "No Platforms found for OpenCL Context" << std::endl;	
	}
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};
	context = clCreateContextFromType(
			contextProperties,
			CL_DEVICE_TYPE_GPU,
			NULL, 
			NULL, 
			&errNum);
	if (errNum != CL_SUCCESS)
	{
		std::cout 
			<< "Could not create GPU context, trying CPU..." 
			<< std::endl;
		context = clCreateContextFromType(
				contextProperties,
				CL_DEVICE_TYPE_CPU,
				NULL, 
				NULL, 
				&errNum);
		if (errNum != CL_SUCCESS)
		{
			std::cerr 
				<< "Failed to create an OpenCL GPU or CPU context."
				<< std::endl;
			return NULL;
		}
	}
	return context;
}	

cl_command_queue DevicePlatformWrapperOpenCL::createCommandQueue(
		cl_context context, 
		cl_device_id *device)
{
	cl_int errNum;
	cl_device_id *devices;
	cl_command_queue commandQueue = NULL;
	size_t deviceBufferSize = -1;

	errNum = clGetContextInfo(
			context, 
			CL_CONTEXT_DEVICES, 
			0, 
			NULL,
			&deviceBufferSize);

	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed call to clGetContextInfo()"
			<< std::endl;
		return NULL;
	}
	if (deviceBufferSize <= 0)
	{
		std::cerr << "No devices available." << std::endl;
		return NULL;
	}

	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(
			context, 
			CL_CONTEXT_DEVICES,	
			deviceBufferSize, 
			devices, 
			NULL);

	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed to get device IDs" << std::endl;
		return NULL;
	}

	commandQueue = clCreateCommandQueue(
			context,
			devices[0],
			0, 
			NULL);
	if (commandQueue == NULL)
	{
		std::cerr << "Failed to create commandQueue for device 0"
			<<std::endl;
		return NULL;
	}
	*device = devices[0];
	delete [] devices;
	return commandQueue;
}

cl_program DevicePlatformWrapperOpenCL::createProgram(cl_context context, cl_device_id device, const char* kernel) 
{
	cl_int errNum;
	cl_program program;
	std::ifstream kernelFile(kernel, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "Failed to open file for reading: " << kernel <<
			std::endl;
		return NULL;
	}
	std::ostringstream oss;
	oss << kernelFile.rdbuf();
	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(context, 1,
			(const char**)&srcStr,
			NULL, NULL);
	if (program == NULL)
	{
		std::cerr << "Failed to create CL program from source." << std::endl;
		return NULL;
	}
	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
				sizeof(buildLog), buildLog, NULL);
		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(program);
		return NULL;
	}
	return program;
}


/**
 * Getter and Setter Section
 **/
cl_context DevicePlatformWrapperOpenCL::getClContext()
{
	return this->context;
}

void DevicePlatformWrapperOpenCL::setClContext(cl_context context)
{
	this->context = context;
}

cl_device_id DevicePlatformWrapperOpenCL::getDeviceId()
{
	return this->device;
}

cl_command_queue DevicePlatformWrapperOpenCL::getClCommandQueueForDevice()
{
	return this->commandQueue;
}	
