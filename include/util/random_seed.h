#include <random>
#include <numeric>
#include <vector>

using namespace std;

/**
 * @brief Generates random seed values for scalar, vector, and matrix data.
 */
class RandomSeed {
	public:
		/**
		 * @brief Generates a random scalar value.
		 * @param expected_value Lower bound or expected reference value for the distribution.
		 * @param standard_deviation Upper bound or spread parameter for the distribution.
		 * @return Generated scalar seed value.
		 */
		float generateRandomScalarSeed(float expected_value, float standard_deviation);

		/**
		 * @brief Generates a random vector seed.
		 * @param num_dimensions Number of vector elements to generate.
		 * @return Generated vector seed values.
		 */
		vector<float> generateRandomVectorSeed(int num_dimensions);

		/**
		 * @brief Generates a random matrix seed.
		 * @param rows Number of matrix rows.
		 * @param columns Number of matrix columns.
		 * @return Generated matrix seed values.
		 */
		vector<vector<float>> generateRandomMatrixSeed(int rows, int columns);
};
