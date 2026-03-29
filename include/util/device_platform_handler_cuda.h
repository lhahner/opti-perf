#include <cuda_runtime.h> 

/**
 * @brief Provides helper access to CUDA device availability and metadata.
 */
class DevicePlatformHandlerCuda {
	private:
		int device_identifier;
		cudaDeviceProp device_propreties;
		char* device_name;	
	
	public:
		/**
		 * @brief Checks whether a CUDA device is available.
		 * @return `true` if a CUDA device is available, otherwise `false`.
		 */
		bool is_device_available(void);
		
		/**
		 * @brief Returns the name of the active CUDA device.
		 * @return Name of the CUDA device.
		 */
		const char* get_device_name(void);
};
