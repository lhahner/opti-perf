#include <gtest/gtest.h>
#include "speedup/linear_operations.h"
#include <vector>

using namespace std;
/**
 * Testing the core functionality.
 **/
TEST(ParallelAdditionTest, ReturnCorrectResult) {
	vector<float> a = {1, 1, 1};
	vector<float> b = {1, 1, 1};
	vector<float> expectedResult = {2, 2, 2};
	LinearOperations linearOperations;
	EXPECT_EQ(linearOperations.add(a, b), expectedResult);
}

/**
 * Testing the core functionality.
 **/
TEST(ParallelSubtractionTest, ReturnCorrectResult) {
	vector<float> a = {1, 1, 1};
	vector<float> b = {1, 1, 1};
	vector<float> expectedResult = {0, 0, 0};
	LinearOperations linearOperations;
	EXPECT_EQ(linearOperations.sub(a, b), expectedResult);
}
