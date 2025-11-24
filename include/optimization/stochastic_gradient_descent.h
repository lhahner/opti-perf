#include <fmt/core.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <random>
#include <iostream>
#include "util/random_seed.h"

using namespace std;

/**
 * Implementation of adam is inspired by the orginal paper and translated from
 * python as given in the source link. 
 * 
 * @Source https://machinelearningmastery.com/adam-optimization-from-scratch/
 **/
class StochasticGradientDescent {
	public:
		float objective(vector<float> postion);
		vector<float> derivative(vector<float> position);
		vector<vector<float>> adam(vector<vector <float>> bounds, int num_iteraions, 
				   float alpha, float beta_1, float beta_2, float epsilon);
	private:
		const unsigned int num_dimensions = 2;
		const unsigned int polynomial = 2;
		RandomSeed randomSeed;
};
