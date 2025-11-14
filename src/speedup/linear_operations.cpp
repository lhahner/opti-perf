#include <stdexcept>
#include <pthread.h>
#include "speedup/linear_operations.h"
#include <vector>

std::vector<float> LinearOperations::add(std::vector<float> a,
					 std::vector<float> b)
{
	std::vector<float> result;
	for (long unsigned int i = 0; i<a.size(); i++) {
		for (long unsigned int j = 0; j<b.size(); j++) {
			result[i] = a[i] + b[j];	
		}
	}
	return result;
}
