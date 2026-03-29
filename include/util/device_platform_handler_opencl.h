#ifndef INCLUDE_UTIL_DEVICE_PLATFORM_HANDLER_OPENCL_CL_
#define INCLUDE_UTIL_DEVICE_PLATFORM_HANDLER_OPENCL_CL_

#include <CL/opencl.hpp>
#include <iostream>
#include <vector>

/**
 * @brief Provides low-level OpenCL platform, device, and context helpers.
 */
class DevicePlatformHandlerOpenCL {
public:
	/**
	 * @brief Checks whether at least one OpenCL platform is available.
	 * @return `true` if an OpenCL platform is available, otherwise `false`.
	 */
	bool isPlatformAvailable(void); 

	/**
	 * @brief Checks whether a GPU device is available for the given platform.
	 * @param platform_id Platform identifier to inspect.
	 * @return `true` if a GPU device is available, otherwise `false`.
	 */
	bool isDeviceAvailable(cl_platform_id platform_id); 

	/**
	 * @brief Creates an OpenCL context for the provided devices and platforms.
	 * @param devices Device identifiers to attach to the context.
	 * @param platformIds Platform identifiers used during context creation.
	 * @return Created OpenCL context.
	 */
	cl_context createAndGetContext(cl_device_id *devices, cl_platform_id* platformIds); 

	/**
	 * @brief Returns the available OpenCL platform identifiers.
	 * @return Pointer to the detected platform identifiers.
	 */
	cl_platform_id* getPlatformIds();

	/**
	 * @brief Returns the GPU device identifiers for the given platform.
	 * @param platformId Platform identifier to query.
	 * @return Pointer to the detected device identifiers.
	 */
	cl_device_id* getDeviceIdsFromPlatformId(cl_platform_id platformId);

	/**
	 * @brief Prints a selected platform information field.
	 * @param id Platform identifier to inspect.
	 * @param name OpenCL platform information selector.
	 * @param str Label used for output.
	 */
	void displayPlatformInfo(cl_platform_id id, cl_platform_info name,
			std::string str);

	/**
	 * @brief Prints summary information about an OpenCL device.
	 * @param deviceId Device identifier to inspect.
	 */
	void displayDeviceInfo(cl_device_id deviceId); 
};
#endif
