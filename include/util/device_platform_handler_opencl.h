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
	bool isPlatformAvailable(void); 
	bool isDeviceAvailable(cl_platform_id platform_id); 
	cl_context createAndGetContext(cl_device_id *devices, cl_platform_id* platformIds); 
	cl_platform_id* getPlatformIds();
	cl_device_id* getDeviceIdsFromPlatformId(cl_platform_id platformId);

	/**
	 * platforms refer to vendors, or rather, vendor OpenCL
	 * runtime drivers. With 1 Intel CPU, 2 Nvidia GPUs and 1
	 * AMD GPU, you will have 3 Platforms, one for Intel, one for
	 * Nvidia and one for AMD. With an AMD CPU and AMD GPU, you
	 * will have a single Platform for both. Same with Intel CPU
	 * and Intel GPU/FPGA, also 1 Platform only.
	 */
	void displayPlatformInfo(cl_platform_id id, cl_platform_info name,
			std::string str);
	/**
	 * In OpenCL devices are considered as the actual GPUs, CPUs or other
	 * which are depended from a platform.
	 */
	void displayDeviceInfo(cl_device_id deviceId); 
};
#endif
