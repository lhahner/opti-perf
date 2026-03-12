#include <benchmark/benchmark.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "benchmark/benchmark_adam_gemm.h"

int main(int argc, char **argv)
{
	const std::filesystem::path config_path =
		std::getenv("OPTI_PERF_CONFIG") != nullptr
			? std::filesystem::path(std::getenv("OPTI_PERF_CONFIG"))
			: std::filesystem::path(OPTI_PERF_SOURCE_DIR) / "config.yaml";

	try
	{
		if (!std::filesystem::exists(config_path))
		{
			std::cerr << "Config file not found: " << config_path << '\n';
			return 1;
		}

		ConfigReader config_reader(config_path.string());
		if (!register_adam_gemm_benchmarks(config_reader))
		{
			const YAML::Node runtime = config_reader.get_runtime_config();
			std::cerr << "No benchmark registered for runtime config: workload="
			          << runtime["workload"].as<std::string>()
			          << ", optimizer=" << runtime["optimizer"].as<std::string>()
			          << ", framework=" << runtime["framework"].as<std::string>() << '\n';
			return 1;
		}
	}
	catch (const std::exception &exception)
	{
		std::cerr << "Failed to register benchmarks from config: " << exception.what() << '\n';
		return 1;
	}

	benchmark::Initialize(&argc, argv);
	if (benchmark::ReportUnrecognizedArguments(argc, argv))
	{
		return 1;
	}

	benchmark::RunSpecifiedBenchmarks();
	benchmark::Shutdown();
	return 0;
}
