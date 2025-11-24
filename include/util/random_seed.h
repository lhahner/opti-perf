#include <random>
#include <numeric>
#include <vector>

using namespace std;

class RandomSeed {
	public:
		float generateRandomScalarSeed(float expected_value, float standard_deviation);
		vector<float> generateRandomVectorSeed(int num_dimensions);
		vector<vector<float>> generateRandomMatrixSeed(int rows, int columns);
};
