#include "util/device_platform_handler_cuda.h"

bool DevicePlatformHandlerCuda::is_device_available(void)
{
	cudaGetDevice(&this->device_identifier);
	cudaGetDeviceProperties(&this->device_propreties, this->device_identifier);
	if (this->device_identifier != 0) {
		return true;
	}
	return false;
}

const char* DevicePlatformHandlerCuda::get_device_name(void)
{
	cudaGetDevice(&this->device_identifier);
	cudaGetDeviceProperties(&this->device_propreties, this->device_identifier);
	this->device_name = this->device_propreties.name;
	return this->device_name;
}

