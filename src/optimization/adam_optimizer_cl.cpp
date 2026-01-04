#include "optimization/adam_optimizer_cl.h"
#include "optimization/optimizer.h"
#include <cstdint>

void AdamOptimizerCl::step(const std::vector<HostParamView>& params, int step_index)
{
	if (!context_ || !queue_ || !kernel_) {
		throw std::runtime_error("AdamOptimizerCl not configured.");
	}

	for (const auto& hp : params) {
		DeviceParamView& dv = toDevice(context_, queue_, hp);
		step_one_tensor(queue_, kernel_, dv, step_index, 
				lr_, beta1_, beta2_, eps_, local_size_);
		fromDevice(queue_, dv, hp);
	}
}

void AdamOptimizerCl::configure(cl_context ctx, cl_command_queue q, cl_kernel k,
		float lr, float b1, float b2, float eps, size_t local_size)
{
	context_ = ctx; queue_ = q; kernel_ = k;
	lr_ = lr; beta1_ = b1; beta2_ = b2; eps_ = eps;
	local_size_ = local_size;
}


void AdamOptimizerCl::step_one_tensor(
		cl_command_queue queue, cl_kernel adam_kernel, DeviceParamView& dv,
		int step_index, float lr, float beta1, float beta2, float eps,
		size_t local_size)
{
	const float b1t = std::pow(beta1, (float)step_index);
	const float b2t = std::pow(beta2, (float)step_index);
	const float bc1 = 1.0f - b1t;
	const float bc2 = 1.0f - b2t;

	if (bc1 == 0.0f || bc2 == 0.0f) {
		throw std::runtime_error("Invalid step_index: bias correction is zero (did you start at step 0?).");
	}

	cl_int err = CL_SUCCESS;
	int arg = 0;

	err = clSetKernelArg(
			adam_kernel, 
			arg++, 
			sizeof(cl_mem), 
			&dv.param);         

	err = clSetKernelArg(adam_kernel, 
			arg++, 
			sizeof(cl_mem), 
			&dv.grad); 

	err = clSetKernelArg(adam_kernel, 
			arg++, 
			sizeof(cl_mem), 
			&dv.m);     

	err = clSetKernelArg(adam_kernel, 
			arg++, 
			sizeof(cl_mem), 
			&dv.v);     

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
			&dv.n);      

	const size_t global_size = ((size_t)dv.n + local_size - 1) / local_size * local_size;
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
	clFinish(queue);
}

DeviceParamView& AdamOptimizerCl::toDevice(
		cl_context ctx,
		cl_command_queue q,
		const HostParamView& hp
		) {
	auto it = device_state_.find(hp.data);

	if (it == device_state_.end()) {
		DeviceParamView dv{};
		dv.n = hp.count;

		cl_int err;
		size_t bytes = dv.n * sizeof(float);

		dv.param = clCreateBuffer(
				ctx, 
				CL_MEM_READ_WRITE, 
				bytes, 
				nullptr, 
				&err);
		if (err != CL_SUCCESS) throw std::runtime_error("param buffer alloc failed");

		dv.grad  = clCreateBuffer(
				ctx, 
				CL_MEM_READ_WRITE, 
				bytes, 
				nullptr, 
				&err);
		if (err != CL_SUCCESS) throw std::runtime_error("grad buffer alloc failed");

		dv.m = clCreateBuffer(
				ctx, 
				CL_MEM_READ_WRITE, 
				bytes, 
				nullptr, 
				&err);
		dv.v = clCreateBuffer(
				ctx, 
				CL_MEM_READ_WRITE, 
				bytes, 
				nullptr, 
				&err);

		float zero = 0.0f;
		clEnqueueFillBuffer(
				q, 
				dv.m, 
				&zero, 
				sizeof(zero), 
				0, 
				bytes, 
				0, 
				nullptr, 
				nullptr);
		clEnqueueFillBuffer(
				q, 
				dv.v, 
				&zero, 
				sizeof(zero), 
				0, 
				bytes, 
				0,
				nullptr, 
				nullptr);
		clFinish(q);

		it = device_state_.emplace(hp.data, dv).first;
	}

	DeviceParamView& dv = it->second;

	if (dv.n != hp.count) {
		throw std::runtime_error("Tensor size changed; resize handling required");
	}

	size_t bytes = dv.n * sizeof(float);
	clEnqueueWriteBuffer(
			q, 
			dv.param, 
			CL_FALSE, 
			0, 
			bytes, 
			hp.data, 
			0, 
			nullptr, 
			nullptr);
	clEnqueueWriteBuffer(q, 
			dv.grad,  
			CL_FALSE, 
			0, 
			bytes, 
			hp.grad, 
			0, 
			nullptr, 
			nullptr);
	return dv;
}

void AdamOptimizerCl::fromDevice(
		cl_command_queue q,
		DeviceParamView& dv,
		const HostParamView& hp
		) {
	const size_t bytes = dv.n * sizeof(float);

	cl_int err = clEnqueueReadBuffer(
			q,
			dv.param,     // device buffer
			CL_TRUE,      // blocking read
			0,
			bytes,
			hp.data,      // host pointer
			0, nullptr, nullptr
			);

	if (err != CL_SUCCESS) {
		throw std::runtime_error("clEnqueueReadBuffer(param -> host) failed");
	}
}
