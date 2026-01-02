#include "optimization/adam_optimizer_cl.h"

void AdamOptimizerCl::step(
		cl_command_queue queue,
		cl_kernel adam_kernel,   
		const std::vector<DeviceTensorAdam>& tensors,
		int step_index, 
		float lr, float beta1, float beta2, float eps,
		size_t local_size)
{
	const float b1t = std::pow(beta1, (float)step_index);
	const float b2t = std::pow(beta2, (float)step_index);
	const float bc1 = 1.0f - b1t;
	const float bc2 = 1.0f - b2t;

	if (bc1 == 0.0f || bc2 == 0.0f) {
		throw std::runtime_error("Invalid step_index: bias correction is zero (did you start at step 0?).");
	}

	for (const auto& t : tensors) {
		if (!t.param || !t.grad || !t.m || !t.v || t.n <= 0) continue;

		cl_int err = CL_SUCCESS;
		int arg = 0;

		err = clSetKernelArg(
				adam_kernel, 
				arg++, 
				sizeof(cl_mem), 
				&t.param);         

		err = clSetKernelArg(adam_kernel, 
				arg++, 
				sizeof(cl_mem), 
				&t.grad); 

		err = clSetKernelArg(adam_kernel, 
				arg++, 
				sizeof(cl_mem), 
				&t.m);     

		err = clSetKernelArg(adam_kernel, 
				arg++, 
				sizeof(cl_mem), 
				&t.v);     

		err = clSetKernelArg(adam_kernel, 
				arg++, 
				sizeof(float), 
				&lr);       

		err = clSetKernelArg(adam_kernel, 
				arg++, 
				sizeof(float), 
				&beta1);    

		err = clSetKernelArg(adam_kernel, 
				arg++, 
				sizeof(float), 
				&beta2);    

		err = clSetKernelArg(adam_kernel, 
				arg++, 
				sizeof(float), 
				&eps);      

		err = clSetKernelArg(adam_kernel, 
				arg++, 
				sizeof(float), 
				&bc1);     

		err = clSetKernelArg(adam_kernel, 
				arg++, 
				sizeof(float), 
				&bc2);      

		err = clSetKernelArg(adam_kernel, 
				arg++, 
				sizeof(int),  
				&t.n);      

		const size_t global_size = ((size_t)t.n + local_size - 1) / local_size * local_size;
		const size_t gws[1] = { global_size };
		const size_t lws[1] = { local_size };

		err = clEnqueueNDRangeKernel(queue, 
				adam_kernel, 
				1, 
				nullptr, 
				gws, 
				lws, 
				0, 
				nullptr, 
				nullptr);
	}
	clFinish(queue);
}
