#include <cuda_runtime.h> 

class DevicePlatformHandlerCuda {
	public:
		bool isDeviceAvailable(void);
	private:
		int deviceIdentifier;
		cudaDeviceProp devicePropreties;
		
};
