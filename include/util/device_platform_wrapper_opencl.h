#include "CL/opencl.h"
#include <CL/cl.h>
#include <stdexcept>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#include "util/benchmark_data.h"
#include "util/setup_wrapper.h"

/**
 * @brief Wrapper to create, set and get OpenCL context, command queue, and program objects.
 * This class abstracts away the details of OpenCL setup and provides a simple interface for 
 * the rest of the codebase to interact with OpenCL resources.
 */
class DevicePlatformWrapperOpenCL {
	public:
		/**
		 * @brief Initialize the OpenCL context, command queue, and program objects. 
		 * This should be called before any OpenCL operations are performed. 
		 * 
		 * @return int is 0 if setup is successful, non-zero otherwise.
		 * The specific error codes can be defined as needed (e.g., SETUP_SUCCESS, SETUP_ERROR, etc.).
		 */
		int setup();

		/**
		 * @brief Get the singleton instance of the DevicePlatformWrapperOpenCL class.
		 * This ensures that there is only one instance managing the OpenCL resources throughout the application.
		 * 
		 * @return DevicePlatformWrapperOpenCL* pointer to the singleton instance.
		 */
		static DevicePlatformWrapperOpenCL* get_instance();
		
		/**
		 * Getter and setter for the context, use create
		 * to allocate the context, get to retrieve the context, 
		 * and set to update the context. 
		 */
		cl_context create_context();
		cl_context get_context();
		void set_context(cl_context);

		/**
		 * Getter and setter for the command queue, use create
		 * to allocate the command queue, get to retrieve the command queue, 
		 * and set to update the command queue. 
		 */
		cl_command_queue create_command_queue(cl_context context, cl_device_id *device);
		cl_command_queue get_command_queue();
		const char *get_device_name() const;

		/**
		 * @brief Create a program object
		 * 
		 * @param context 
		 * @param device 
		 * @param kernel 
		 * @return cl_program 
		 */
		cl_program create_program(cl_context context, cl_device_id device, const char* kernel);
		
		/**
		 * @brief Get the device id object
		 * 
		 * @return cl_device_id 
		 */
		cl_device_id get_device_id();
	
	private:
		static DevicePlatformWrapperOpenCL* device_platform_wrapper_OpenCL;

		const char* kernel_name;
		const char* kernel_file;
		
		size_t device_buffer_size = -1;
		
		cl_context context = 0;
		
		cl_command_queue command_queue = 0;
		
		cl_program program = 0;
		
		cl_device_id device = 0;
		std::string device_name;
		
		cl_kernel kernel = 0;
		
		cl_mem memory_objects[N_MEMOBJ] = {};
		
		cl_int error_number;
};
