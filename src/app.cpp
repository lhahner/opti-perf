#include <iostream>

#include "../include/util/device_platform_handler_opencl.h"
#include "benchmark/workloads/machinelearning/mnist.h"

int main(int argc, char** argv) {
	DevicePlatformHandlerOpenCL* devicePlatformHandlerOpenCL = new DevicePlatformHandlerOpenCL();
	devicePlatformHandlerOpenCL->isPlatformAvailable();
	delete devicePlatformHandlerOpenCL;
    return 0;
}
