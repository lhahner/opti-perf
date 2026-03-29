#include "optimization/adam_optimizer_cl.h"
#include "optimization/optimizer.h"
#include <cstdint>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

namespace {
constexpr int kWarmupSteps = 10;
constexpr const char *kValidationLogFile = "validation-benchmark-logs.csv";

const char *format_timestamp()
{
	auto now = std::chrono::system_clock::now();
	std::time_t tt = std::chrono::system_clock::to_time_t(now);
	std::tm tm{};
	localtime_r(&tt, &tm);
	std::ostringstream ts;
	ts << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
	static thread_local std::string ts_str;
	ts_str = ts.str();
	return ts_str.c_str();
}
} // namespace

void AdamOptimizerCl::step(BenchmarkData *benchmarkData, const std::vector<HostParamView> &params, int step_index)
{
	if (!context_ || !queue_ || !kernel_)
	{
		throw std::runtime_error("AdamOptimizerCl not configured.");
	}

	double total_transfer_ms = 0.0;
	double total_compute_ms = 0.0;
	double total_d2h_ms = 0.0;
	for (const auto &hp : params)
	{
		DeviceParamView &dv = toDevice(benchmarkData, context_, queue_, hp);
		total_compute_ms += step_one_tensor(queue_, kernel_, dv, step_index,
											lr_, beta1_, beta2_, eps_, local_size_);
		total_d2h_ms += fromDevice(queue_, dv, hp);
		total_transfer_ms += static_cast<double>(benchmarkData->time_ms);
	}

	const bool use_warmup =
		benchmarkData != nullptr && benchmarkData->log_filename == kValidationLogFile;

	if (benchmarkData != nullptr && (!use_warmup || step_index > kWarmupSteps))
	{
		benchmarkData->timestamp = format_timestamp();
		benchmarkData->workload_type = "h2d_transfer";
		benchmarkData->time_ms = static_cast<float>(total_transfer_ms);
		logger.logToCsv(*benchmarkData, benchmarkData->log_filename.c_str());

		benchmarkData->timestamp = format_timestamp();
		benchmarkData->workload_type = "compute";
		benchmarkData->time_ms = static_cast<float>(total_compute_ms);
		logger.logToCsv(*benchmarkData, benchmarkData->log_filename.c_str());

		benchmarkData->timestamp = format_timestamp();
		benchmarkData->workload_type = "d2h_transfer";
		benchmarkData->time_ms = static_cast<float>(total_d2h_ms);
		logger.logToCsv(*benchmarkData, benchmarkData->log_filename.c_str());
	}
}

void AdamOptimizerCl::configure(cl_context ctx, cl_command_queue q, cl_kernel k,
								float lr, float b1, float b2, float eps, size_t local_size)
{
	context_ = ctx;
	queue_ = q;
	kernel_ = k;
	lr_ = lr;
	beta1_ = b1;
	beta2_ = b2;
	eps_ = eps;
	local_size_ = local_size;
}

double AdamOptimizerCl::step_one_tensor(
	cl_command_queue queue, cl_kernel adam_kernel, DeviceParamView &dv,
	int step_index, float lr, float beta1, float beta2, float eps,
	size_t local_size)
{
	const float b1t = std::pow(beta1, (float)step_index);
	const float b2t = std::pow(beta2, (float)step_index);
	const float bc1 = 1.0f - b1t;
	const float bc2 = 1.0f - b2t;

	if (bc1 == 0.0f || bc2 == 0.0f)
	{
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
	const size_t gws[1] = {global_size};
	const size_t lws[1] = {local_size};

	cl_event event;
	err = clEnqueueNDRangeKernel(queue,
								 adam_kernel,
								 1,
								 nullptr,
								 gws,
								 lws,
								 0,
								 nullptr,
								 &event);
	clWaitForEvents(1, &event);
	clFinish(queue);
	cl_ulong time_start;
	cl_ulong time_end;

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

	double nanoSeconds = time_end - time_start;
	std::cout << toString(Marker::INFO) << "OpenCl Execution time is: " << (nanoSeconds / 1000000.0) << " milliseconds \n";
	clReleaseEvent(event);
	return nanoSeconds / 1000000.0;
}

DeviceParamView &AdamOptimizerCl::toDevice(
	BenchmarkData *benchmarkData,
	cl_context ctx,
	cl_command_queue q,
	const HostParamView &hp)
{
	auto it = device_state_.find(hp.data);

	if (it == device_state_.end())
	{
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
		if (err != CL_SUCCESS)
			throw std::runtime_error("param buffer alloc failed");

		dv.grad = clCreateBuffer(
			ctx,
			CL_MEM_READ_WRITE,
			bytes,
			nullptr,
			&err);
		if (err != CL_SUCCESS)
			throw std::runtime_error("grad buffer alloc failed");

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

	DeviceParamView &dv = it->second;

	if (dv.n != hp.count)
	{
		throw std::runtime_error("Tensor size changed; resize handling required");
	}

	size_t bytes = dv.n * sizeof(float);
	cl_event transferEventParameter, transferEventGradient;
	cl_int errParam = clEnqueueWriteBuffer(
		q,
		dv.param,
		CL_FALSE,
		0,
		bytes,
		hp.data,
		0,
		nullptr,
		&transferEventParameter);
	if (errParam != CL_SUCCESS)
	{
		throw std::runtime_error("clEnqueueWriteBuffer(param) failed");
	}
	clWaitForEvents(1, &transferEventParameter);
	cl_ulong startParam = 0;
	cl_ulong endParam = 0;
	clGetEventProfilingInfo(transferEventParameter, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startParam, NULL);
	clGetEventProfilingInfo(transferEventParameter, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endParam, NULL);
	cl_ulong transferTimeParam = endParam - startParam;
	std::cout << toString(Marker::INFO) << "OpenCl Transfer to Device time Parameter is: " << (transferTimeParam / 1000000.0) << " milliseconds \n";
	cl_int errGrad = clEnqueueWriteBuffer(q,
										  dv.grad,
										  CL_FALSE,
										  0,
										  bytes,
										  hp.grad,
										  0,
										  nullptr,
										  &transferEventGradient);
	if (errGrad != CL_SUCCESS)
	{
		throw std::runtime_error("clEnqueueWriteBuffer(grad) failed");
	}
	clWaitForEvents(1, &transferEventGradient);
	unsigned long startGrad = 0;
	unsigned long endGrad = 0;
	clGetEventProfilingInfo(transferEventGradient, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startGrad, NULL);
	clGetEventProfilingInfo(transferEventGradient, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endGrad, NULL);
	unsigned long transferTimeGrad = endGrad - startGrad + transferTimeParam;

	if (benchmarkData != nullptr)
	{
		benchmarkData->time_ms = static_cast<float>(transferTimeGrad / 1000000.0);
	}
	std::cout << toString(Marker::INFO) << "OpenCl total transfer to device time with gradient is: " << (transferTimeGrad / 1000000.0) << " milliseconds \n";
	clReleaseEvent(transferEventParameter);
	clReleaseEvent(transferEventGradient);
	return dv;
}

double AdamOptimizerCl::fromDevice(
	cl_command_queue q,
	DeviceParamView &dv,
	const HostParamView &hp)
{
	const size_t bytes = dv.n * sizeof(float);
	cl_event read_event = nullptr;

	cl_int err = clEnqueueReadBuffer(
		q,
		dv.param, // device buffer
		CL_FALSE,
		0,
		bytes,
		hp.data, // host pointer
		0, nullptr, &read_event);

	if (err != CL_SUCCESS)
	{
		throw std::runtime_error("clEnqueueReadBuffer(param -> host) failed");
	}
	clWaitForEvents(1, &read_event);

	cl_ulong start = 0;
	cl_ulong end = 0;
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	clReleaseEvent(read_event);
	return static_cast<double>(end - start) / 1000000.0;
}
