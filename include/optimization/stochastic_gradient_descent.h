#include <vector>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <random>

using namespace std;

class StochasticGradientDescent {
	public:
		float objective(vector<float> postion);
		vector<float> derivative(vector<float> position);
		vector<vector<float>> adam(vector<vector <float>> bounds, int num_iteraions, 
				   float alpha, float beta_1, float beta_2, float epsilon);
	private:
		const unsigned int num_dimensions = 2;
		const unsigned int polynomial = 2;
};
