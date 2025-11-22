#include <gradient_descent.h>
#include <vector>

using namespace std;

class StochasticGradientDescent : public GradientDescent {
	public:
		vector<float> objective(vector<float> postion);
		vector<float> derivative(vector<float> position);
		vector<float> adam(vector<float> seed, int num_iteraions, 
				   float alpha, float beta_1, float beta_2, float epsilon);
};
