#include <CL/cl.h>
#include <benchmark/benchmark.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include "benchmark/benchmark_adam_training.h"
#include "benchmark/workloads/mnist_linear/mnist_linear.h"
#include "optimization/adam_optimizer.h"
#include "optimization/adam_optimizer_cl.h"
#include "optimization/adam_optimizer_cu.h"
#include "util/device_platform_wrapper_opencl.h"
#include "util/logger.h"

#ifndef OPTI_PERF_SOURCE_DIR
#define OPTI_PERF_SOURCE_DIR "."
#endif

namespace
{
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

struct AdamTrainingConfig
{
	float learning_rate = 1e-3f;
	float beta1 = 0.9f;
	float beta2 = 0.999f;
	float epsilon = 1e-8f;
	int batch_size = 64;
	int max_samples = 1024;
	std::string framework = "CPU";
	std::string dataset_dir = std::string(OPTI_PERF_SOURCE_DIR) + "/data/mnist";
};

AdamTrainingConfig g_training_config;

const AdamTrainingConfig &config()
{
	return g_training_config;
}

AdamTrainingConfig load_config(ConfigReader &config_reader)
{
	const YAML::Node runtime = config_reader.get_runtime_config();
	const YAML::Node optimizer = config_reader.get_optimizer_config();
	const YAML::Node workload = config_reader.get_workload_config();

	AdamTrainingConfig config;
	config.framework = runtime["framework"].as<std::string>();
	config.learning_rate = optimizer["learning_rate"].as<float>();
	config.beta1 = optimizer["beta_1"].as<float>();
	config.beta2 = optimizer["beta_2"].as<float>();
	config.epsilon = optimizer["epsilon"].as<float>();
	config.batch_size = optimizer["batch_size"].as<int>();

	if (workload["dataset_dir"])
	{
		config.dataset_dir = workload["dataset_dir"].as<std::string>();
	}
	if (workload["max_samples"])
	{
		config.max_samples = workload["max_samples"].as<int>();
	}

	return config;
}

BenchmarkData make_benchmark_data(const AdamTrainingConfig &cfg, const char *framework, const char *device)
{
	return BenchmarkData(
		nullptr,
		"",
		framework,
		(char *)"MNIST_Training_Adam",
		(char *)"training_step",
		device,
		cfg.batch_size,
		static_cast<long>(cfg.max_samples),
		(char *)"Adam",
		cfg.learning_rate,
		cfg.beta1,
		cfg.beta2,
		cfg.epsilon,
		0.0f,
		0,
		0.0f,
		0.0f);
}

void log_evaluation(BenchmarkData &benchmark_data, MnistLinear &workload)
{
	Logger logger;
	benchmark_data.timestamp = format_timestamp();
	benchmark_data.workload_type = "evaluation";
	benchmark_data.loss = workload.evaluateTestLoss();
	benchmark_data.accuracy = workload.evaluateTestAccuracy();
	benchmark_data.time_ms = 0.0f;
	logger.logToCsv(benchmark_data, "benchmarks-logs.csv");
}
} // namespace

static void BM_Training_Adam(benchmark::State &state)
{
	const AdamTrainingConfig &cfg = config();
	BenchmarkData benchmark_data = make_benchmark_data(cfg, "CPU", "CPU");
	MnistLinear workload(cfg.dataset_dir, cfg.batch_size, cfg.max_samples);
	AdamOptimizer adam(cfg.learning_rate, cfg.beta1, cfg.beta2, cfg.epsilon);

	for (auto _ : state)
	{
		for (int t = 1; t <= state.range(0); ++t)
		{
			benchmark_data.batch_index = t;
			workload.runForward();
			benchmark_data.loss = workload.computeLoss().second;
			adam.step(&benchmark_data, workload.parameters(), t);
			benchmark::DoNotOptimize(benchmark_data.loss);
		}
		log_evaluation(benchmark_data, workload);
	}
}

static void BM_Training_Adam_cl(benchmark::State &state)
{
	const AdamTrainingConfig &cfg = config();
	BenchmarkData benchmark_data = make_benchmark_data(cfg, "OpenCL", "GPU");
	MnistLinear workload(cfg.dataset_dir, cfg.batch_size, cfg.max_samples);
	AdamOptimizerCl adam(cfg.learning_rate, cfg.beta1, cfg.beta2, cfg.epsilon);

	auto *wrapper = DevicePlatformWrapperOpenCL::get_instance();
	int setup_success = wrapper->setup();
	if (setup_success != SETUP_SUCCESS)
	{
		throw std::runtime_error("OpenCL setup initialization failed");
	}
	benchmark_data.device_name = wrapper->get_device_name();

	cl_context ctx = wrapper->get_context();
	cl_command_queue queue = wrapper->get_command_queue();

	static const char kernel_path[] = OPTI_PERF_SOURCE_DIR "/kernels/adam_optimizer.cl";
	cl_program program = wrapper->create_program(ctx, wrapper->get_device_id(), kernel_path);

	cl_int err = CL_SUCCESS;
	cl_kernel kernel = clCreateKernel(program, "adam_update", &err);
	if (err != CL_SUCCESS)
	{
		clReleaseProgram(program);
		throw std::runtime_error("clCreateKernel(adam_update) failed");
	}

	adam.configure(ctx, queue, kernel, cfg.learning_rate, cfg.beta1, cfg.beta2, cfg.epsilon, 256);

	for (auto _ : state)
	{
		for (int t = 1; t <= state.range(0); ++t)
		{
			benchmark_data.batch_index = t;
			workload.runForward();
			benchmark_data.loss = workload.computeLoss().second;
			adam.step(&benchmark_data, workload.parameters(), t);
			benchmark::DoNotOptimize(benchmark_data.loss);
		}
		log_evaluation(benchmark_data, workload);
	}

	clReleaseKernel(kernel);
	clReleaseProgram(program);
}

static void BM_Training_Adam_cuda(benchmark::State &state)
{
	const AdamTrainingConfig &cfg = config();
	BenchmarkData benchmark_data = make_benchmark_data(cfg, "CUDA", "GPU");
	MnistLinear workload(cfg.dataset_dir, cfg.batch_size, cfg.max_samples);
	AdamOptimizerCu adam(cfg.learning_rate, cfg.beta1, cfg.beta2, cfg.epsilon);

	for (auto _ : state)
	{
		for (int t = 1; t <= state.range(0); ++t)
		{
			benchmark_data.batch_index = t;
			workload.runForward();
			benchmark_data.loss = workload.computeLoss().second;
			adam.step(&benchmark_data, workload.parameters(), t);
			benchmark::DoNotOptimize(benchmark_data.loss);
		}
		log_evaluation(benchmark_data, workload);
	}
}

bool register_adam_training_benchmarks(ConfigReader &config_reader)
{
	const YAML::Node runtime = config_reader.get_runtime_config();
	const std::string workload = runtime["workload"].as<std::string>();
	const std::string optimizer = runtime["optimizer"].as<std::string>();

	if (workload != "Training" || optimizer != "Adam")
	{
		return false;
	}

	g_training_config = load_config(config_reader);

	if (g_training_config.framework == "CPU" || g_training_config.framework == "None")
	{
		benchmark::RegisterBenchmark("BM_Training_Adam", &BM_Training_Adam)->Arg(10)->Iterations(1);
		return true;
	}
	if (g_training_config.framework == "OpenCL")
	{
		benchmark::RegisterBenchmark("BM_Training_Adam_cl", &BM_Training_Adam_cl)->Arg(10)->Iterations(1);
		return true;
	}
	if (g_training_config.framework == "CUDA")
	{
		benchmark::RegisterBenchmark("BM_Training_Adam_cuda", &BM_Training_Adam_cuda)->Arg(10)->Iterations(1);
		return true;
	}

	throw std::runtime_error("Unsupported framework in runtime config: " + g_training_config.framework);
}
