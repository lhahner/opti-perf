#include <cuda_runtime.h> 

class DevicePlatformHandlerCuda {
	private:
		int device_identifier;
		cudaDeviceProp device_propreties;
		char* device_name;	
	
	public:
		/**
		 * @brief Checks if a CUDA device is available
		 * 
		 * @return true if a CUDA device is available, false otherwise
		 */
		bool is_device_available(void);
		
		/**
		 * @brief Get the name of the CUDA device
		 * 
		 * @return const char* name of the CUDA device
		 */
		const char* get_device_name(void);
};
