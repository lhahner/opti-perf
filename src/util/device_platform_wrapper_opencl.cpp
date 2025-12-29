#include "util/device_platform_wrapper_opencl.h"
#include <CL/cl.h>
#include <stdexcept>

/**
 * Core Setup and Implementation
 *
 * @return 0 if success and 1 if not, logs failure cause anyway.
 **/
int DevicePlatformWrapperOpenCL::initSetup() 
{
	this->context = this->createContext();
	if (this->getClContext() == NULL) {
		std::cerr << "Context Creation in inital setup for OpenCL failed." << std::endl;
		return SETUP_FAILURE;	
	}
	return SETUP_SUCCESS;
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
	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);

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
		std::cout << "Could not create GPU context, trying CPU..." << std::endl;
		context = clCreateContextFromType(contextProperties,
						  CL_DEVICE_TYPE_CPU,
					          NULL, 
						  NULL, 
						  &errNum);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "Failed to create an OpenCL GPU or CPU context.";
			return NULL;
		}
	}
	return context;
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
