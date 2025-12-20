#include <benchmark/benchmark.h>
#include "benchmark/workloads/machinelearning/mnist.h"
#include "optimization/stochastic_gradient_descent.h"

static void BM_load_batches(benchmark::State &state) {
	for (auto _ : state) {
		Mnist mnist;
		mnist.load_batches();
	}
}
BENCHMARK(BM_load_batches);

static void BM_optimize(benchmark::State &state) {
	for (auto _ : state) {
		StochasticGradientDescent stochasticGradientDescent;
		stochasticGradientDescent.optimize();
	}
}
BENCHMARK(BM_load_batches);

