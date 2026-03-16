#include "util/device_platform_wrapper_opencl.h"
#include <CL/cl.h>
#include <stdexcept>

DevicePlatformWrapperOpenCL *DevicePlatformWrapperOpenCL::device_platform_wrapper_OpenCL = nullptr;
;

/**
 * Core Setup and Implementation
 *
 * @return 0 if success and 1 if not, logs failure cause anyway.
 **/
int DevicePlatformWrapperOpenCL::setup()
{
	this->context = this->create_context();
	if (this->get_context() == NULL)
	{
		std::cerr
			<< "Context Creation in inital setup for OpenCL failed."
			<< std::endl;
		return SETUP_FAILURE;
	}

	this->command_queue = this->create_command_queue(
		this->context,
		&this->device);

	if (this->command_queue == NULL)
	{
		std::cerr << "Failed to created commandQueue in inital seutp."
				  << std::endl;
		return SETUP_FAILURE;
	}

	return SETUP_SUCCESS;
}

DevicePlatformWrapperOpenCL *DevicePlatformWrapperOpenCL::get_instance()
{
	if (device_platform_wrapper_OpenCL == nullptr)
	{
		device_platform_wrapper_OpenCL =
			new DevicePlatformWrapperOpenCL();
		return device_platform_wrapper_OpenCL;
	}
	return device_platform_wrapper_OpenCL;
}

cl_context DevicePlatformWrapperOpenCL::create_context()
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
			0};
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

cl_command_queue DevicePlatformWrapperOpenCL::create_command_queue(cl_context context, cl_device_id *device)
{
	cl_int error_number;
	cl_device_id *devices;
	cl_command_queue command_queue = NULL;
	size_t device_buffer_size = -1;

	error_number = clGetContextInfo(
		context,
		CL_CONTEXT_DEVICES,
		0,
		NULL,
		&device_buffer_size);

	if (error_number != CL_SUCCESS)
	{
		std::cerr << "Failed call to clGetContextInfo()"
				  << std::endl;
		return NULL;
	}
	if (device_buffer_size <= 0)
	{
		std::cerr << "No devices available." << std::endl;
		return NULL;
	}

	devices = new cl_device_id[device_buffer_size / sizeof(cl_device_id)];
	error_number = clGetContextInfo(
		context,
		CL_CONTEXT_DEVICES,
		device_buffer_size,
		devices,
		NULL);

	if (error_number != CL_SUCCESS)
	{
		std::cerr << "Failed to get device IDs" << std::endl;
		return NULL;
	}

	command_queue = clCreateCommandQueue(
		context,
		devices[0],
		CL_QUEUE_PROFILING_ENABLE,
		NULL);
	if (command_queue == NULL)
	{
		std::cerr << "Failed to create commandQueue for device 0"
				  << std::endl;
		return NULL;
	}
	*device = devices[0];

	size_t name_size = 0;
	cl_int err = clGetDeviceInfo(
		*device,
		CL_DEVICE_NAME,
		0,
		nullptr,
		&name_size);

	if (err != CL_SUCCESS)
	{
		std::cerr << "Failed to get device name size\n";
		delete[] devices;
		return NULL;
	}

	std::vector<char> opencl_device_name(name_size);
	err = clGetDeviceInfo(
		*device,
		CL_DEVICE_NAME,
		name_size,
		opencl_device_name.data(),
		nullptr);

	if (err != CL_SUCCESS)
	{
		std::cerr << "Failed to get device name\n";
		delete[] devices;
		return NULL;
	}

	device_name = opencl_device_name.data();
	std::cout << "Using OpenCL device: " << device_name << std::endl;

	delete[] devices;
	return command_queue;
}

cl_program DevicePlatformWrapperOpenCL::create_program(cl_context context, cl_device_id device, const char *kernel)
{
	cl_int errNum;
	cl_program program;
	std::ifstream kernelFile(kernel, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "Failed to open file for reading: " << kernel << std::endl;
		return NULL;
	}
	std::ostringstream oss;
	oss << kernelFile.rdbuf();
	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(context, 1,
										(const char **)&srcStr,
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
cl_context DevicePlatformWrapperOpenCL::get_context()
{
	return this->context;
}

void DevicePlatformWrapperOpenCL::set_context(cl_context context)
{
	this->context = context;
}

cl_device_id DevicePlatformWrapperOpenCL::get_device_id()
{
	return this->device;
}

cl_command_queue DevicePlatformWrapperOpenCL::get_command_queue()
{
	return this->command_queue;
}

const char *DevicePlatformWrapperOpenCL::get_device_name() const
{
	return this->device_name.c_str();
}
