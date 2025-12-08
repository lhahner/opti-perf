#include "optimization/stochastic_gradient_descent.h"

float StochasticGradientDescent::objective(vector<float> position) 
{
	if (position.size() > num_dimensions) {
		throw length_error("The given position exceed the limited dimensions");
	}
	vector<float> results(num_dimensions);
	for (auto position_at_dim : position) {
		results.push_back(pow(position_at_dim, polynomial));
	}
	return accumulate(results.begin(), results.end(), 0);
}	

vector<float> StochasticGradientDescent::derivative(vector<float> position) {
	if (position.size() != num_dimensions) {
		throw length_error("The given position exceed the limited dimensions.");
	}
	vector<float> results(num_dimensions);
	for (auto position_at_dim : position) {
		results.push_back(position_at_dim * polynomial);
	}
	return results;
}

vector<vector<float>> StochasticGradientDescent::adam(vector<vector <float>> bounds, int num_iterations, float alpha, 
				              float beta_1, float beta_2, float epsilon) 
{
	if ((bounds.size() != num_dimensions) || bounds.size() == 0) {
		throw length_error("The given bounds vector is empty or not equal size as dimensions.");
	} else if (bounds[0].size() > 2 || bounds[0].size() < 2) {
		throw length_error("The given bounds vectors range is larger then 2 and cannot be computed.");
	}
	
	vector<float> input(num_dimensions);
	for (int i = 0; i<bounds.size(); i++) {
		float random_decimal = randomSeed.generateRandomScalarSeed(2.0, 4.9)  / 10;
		float lower_bound = bounds[i][0];
		float upper_bound = bounds[i][1];

		input[i] = lower_bound + random_decimal * (lower_bound - upper_bound); // Here we could use our parallel function
	}

	float score = this->objective(input);
	vector<float> first_moment(bounds.size(), 0);
	vector<float> second_moment(bounds.size(), 0);
 	vector<float> gradient;	
	for (int i = 0; i<num_iterations; i++) {
		gradient = this->derivative(bounds[i]);	
		for (int j = 0; i<num_dimensions;i++) {
			first_moment[j] = beta_1 * first_moment[j] + (1.0 - beta_1) * gradient[j];
			second_moment[i] = beta_2 * second_moment[j] + (1.0 - beta_2) * pow(gradient[j],2);
			float first_moment_hat = first_moment[j] / (1.0 - pow(beta_1, i+1));
			float second_moment_hat = second_moment[j] / (1.0 - pow(beta_2, i+1));
			input[j] = input[j] - alpha * first_moment_hat / (sqrt(second_moment_hat) + epsilon);
		}
	score = objective(input);
	}
	return {input, {score}};
}
