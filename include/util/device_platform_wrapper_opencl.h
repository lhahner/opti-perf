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
 * @brief Wraps OpenCL platform and device resource initialization.
 *
 * This class abstracts OpenCL context, queue, program, and device access for
 * the rest of the codebase.
 */
class DevicePlatformWrapperOpenCL {
	public:
		/**
		 * @brief Initializes the OpenCL context and command queue.
		 * @return `SETUP_SUCCESS` on success, otherwise `SETUP_FAILURE`.
		 */
		int setup();

		/**
		 * @brief Returns the singleton wrapper instance.
		 * @return Pointer to the shared wrapper instance.
		 */
		static DevicePlatformWrapperOpenCL* get_instance();
		
		/**
		 * @brief Creates an OpenCL context for the selected platform.
		 * @return Created OpenCL context or `NULL` on failure.
		 */
		cl_context create_context();

		/**
		 * @brief Returns the stored OpenCL context.
		 * @return Current OpenCL context.
		 */
		cl_context get_context();

		/**
		 * @brief Stores the OpenCL context managed by the wrapper.
		 * @param context OpenCL context to store.
		 */
		void set_context(cl_context);

		/**
		 * @brief Creates a command queue for the first device in the context.
		 * @param context OpenCL context containing the target device.
		 * @param device Output parameter receiving the selected device id.
		 * @return Created command queue or `NULL` on failure.
		 */
		cl_command_queue create_command_queue(cl_context context, cl_device_id *device);

		/**
		 * @brief Returns the stored OpenCL command queue.
		 * @return Current command queue.
		 */
		cl_command_queue get_command_queue();

		/**
		 * @brief Returns the selected device name.
		 * @return Null-terminated device name string.
		 */
		const char *get_device_name() const;

		/**
		 * @brief Creates and builds an OpenCL program from a kernel source file.
		 * @param context OpenCL context used to create the program.
		 * @param device OpenCL device used to build the program.
		 * @param kernel Path to the kernel source file.
		 * @return Built OpenCL program or `NULL` on failure.
		 */
		cl_program create_program(cl_context context, cl_device_id device, const char* kernel);
		
		/**
		 * @brief Returns the stored OpenCL device id.
		 * @return Current device id.
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
