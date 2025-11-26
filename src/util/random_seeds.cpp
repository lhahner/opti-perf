#include "util/random_seed.h"
#include <random>

float RandomSeed::generateRandomScalarSeed(float expected_value, float standard_deviation) {
	mt19937 gen;
	uniform_real_distribution<float> distribution(expected_value, standard_deviation);
	return distribution(gen);
}
