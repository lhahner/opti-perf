#ifndef INCLUDE_UTIL_DEVICE_PLATFORM_HANDLER_OPENCL_CL_
#define INCLUDE_UTIL_DEVICE_PLATFORM_HANDLER_OPENCL_CL_

#include <CL/opencl.hpp>
#include <iostream>
#include <vector>

/**
 * This should be a wrapper-class to interact from host to device/kernel
 * and backwards using OpenCL. 
 **/
class DevicePlatformHandlerOpenCL {
public:
	bool isPlatformAvailable(void) {
		cl_int errNum;
		cl_uint numPlatforms;
		cl_platform_id *platformIds;
		cl_context context = NULL;
		errNum = clGetPlatformIDs(0, NULL, &numPlatforms);

		if (errNum != CL_SUCCESS || numPlatforms <= 0) {
			std::cerr << "Failed to find any OpenCL platform." << std::endl;
			return false;
		}
		std::cout << "Found OpenCL platform." << std::endl;
		return true;
	}

	bool isDeviceAvailable(cl_platform_id platform_id) {
		cl_int errNum;
		cl_uint numDevices;
		cl_device_id deviceIds[1];
		errNum = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL,
				&numDevices);

		if (numDevices < 1 || errNum != CL_SUCCESS) {
			std::cout << "No GPU device found for platform " << platform_id
					<< std::endl;
			return false;
		}
		return true;
	}

	cl_context createAndGetContext(cl_device_id *devices, cl_platform_id* platformIds) {
		cl_uint num;
		cl_context context;
		size_t size;

		cl_context_properties properties[] = {
				CL_CONTEXT_PLATFORM, (cl_context_properties) platformIds, 0
		};

		return clCreateContext(properties, size / sizeof(cl_device_id),
				devices,
				NULL,
				NULL,
				NULL);;
	}

	cl_platform_id* getPlatformIds() {
		cl_uint numPlatforms;
		cl_platform_id *platformIDs;
		cl_context context = NULL;
		size_t size;

		clGetPlatformIDs(0, NULL, &numPlatforms);
		platformIDs = (cl_platform_id*) alloca(
				sizeof(cl_platform_id) * numPlatforms);
		clGetPlatformIDs(numPlatforms, platformIDs, NULL);

		return platformIDs;
	}

	cl_device_id* getDeviceIdsFromPlatformId(cl_platform_id platformId)
	{
		cl_uint num;
		cl_device_id *devices;
		cl_context context;
		size_t size;

		clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 0, 
				NULL, &num);
		if (num > 0) {
			devices = (cl_device_id*) alloca(num);
			clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 
					num, devices, NULL);
		}
		return devices;
	}

	/**
	 * platforms refer to vendors, or rather, vendor OpenCL
	 * runtime drivers. With 1 Intel CPU, 2 Nvidia GPUs and 1
	 * AMD GPU, you will have 3 Platforms, one for Intel, one for
	 * Nvidia and one for AMD. With an AMD CPU and AMD GPU, you
	 * will have a single Platform for both. Same with Intel CPU
	 * and Intel GPU/FPGA, also 1 Platform only.
	 */
	void displayPlatformInfo(cl_platform_id id, cl_platform_info name,
			std::string str) {
		cl_int errNum;
		std::size_t paramValueSize;
		errNum = clGetPlatformInfo(id, name, 0,
		NULL, &paramValueSize);
		if (errNum != CL_SUCCESS) {
			std::cerr << "Failed to find OpenCL platform " << str << "."
					<< std::endl;
			return;
		}
		char *info = (char*) alloca(sizeof(char) * paramValueSize);
		errNum = clGetPlatformInfo(id, name, paramValueSize, info,
		NULL);
		if (errNum != CL_SUCCESS) {
			std::cerr << "Failed to find OpenCL platform " << str << "."
					<< std::endl;
			return;
		}
		std::cout << "\t" << str << ":\t" << info << std::endl;
	}

	/**
	 * In OpenCL devices are considered as the actual GPUs, CPUs or other
	 * which are depended from a platform.
	 */
	void displayDeviceInfo(cl_device_id deviceId) {
		cl_int err;
		size_t size;
		cl_int maxComputeUnits;
		err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_COMPUTE_UNITS,
				sizeof(cl_uint), &maxComputeUnits, &size);
		std::cout << "Device has max compute units: " << maxComputeUnits
				<< std::endl;
	}
};

#endif
