#include <cstdlib>
#include <iostream>
#include "speedup/linear_operations.h"
#include <vector>

int main()
{
	LinearOperations* lo = new LinearOperations;
	std::vector<float> vec = {1, 2};
	std::vector<float> vec1 = {1, 2};
	std::vector<float> res = lo->add(vec, vec1);
	for (int i = 0; i<res.size(); i++)
		std::cout << "Result: " << res[i] << "."; 
}
