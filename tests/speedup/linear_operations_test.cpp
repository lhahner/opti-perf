#include <gtest/gtest.h>
#include "speedup/linear_operations.h"
#include <vector>

/**
 * Testing the core functionality.
 **/
TEST(ParallelAdditionTest, ReturnCorrectResult) {
	std::vector<float> a = {1, 1, 1};
	std::vector<float> b = {1, 1, 1};
	std::vector<float> expectedResult = {2, 2, 2};
	LinearOperations linearOperations;
	EXPECT_EQ(linearOperations.add(a, b), expectedResult);
}
