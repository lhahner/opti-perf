#include "util/device_platform_handler_cuda.h"

bool DevicePlatformHandlerCuda::isDeviceAvailable(void)
{
	cudaGetDevice(&this->deviceIdentifier);
	cudaGetDeviceProperties(&this->devicePropreties, this->deviceIdentifier);
	if (this->deviceIdentifier != 0) {
		return true;
	}
	return false;
}

