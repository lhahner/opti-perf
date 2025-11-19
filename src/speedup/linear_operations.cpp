#include <stdexcept>
#include <pthread.h>
#include "speedup/linear_operations.h"
#include <vector>
#include <stdexcept>
#include <thread>

int thread_count;

enum Operation {
    ADD,
    SUB,
    MULT,
    DIV,
    TRANS,
    INVERSE
};

std::vector<float> LinearOperations::add(std::vector<float> a,
					 std::vector<float> b)
{
    if (a.size() == 0 && b.size() == 0) {
        std::runtime_error("a or b is null and therefore no calculation is possible");
    }
    std::vector<float> result;
#pragma omp parallel 
    {
	# pragma omp parallel for 
	for (long unsigned int i = 0; i<a.size(); i++) {
		for (long unsigned int j = 0; j<b.size(); j++) {
			result[i] = a[i] + b[j];	
		}
	}
    }
	return result;
}

std::vector<float> LinearOperations::sub(std::vector<float> a, std::vector<float> b)
{
    if (a.size() == 0 && b.size() == 0) {
        std::runtime_error("a or b is null and therefore no calculation is possible");
    }
    std::vector<float> result;
	for (long unsigned int i = 0; i<a.size(); i++) {
		for (long unsigned int j = 0; j<b.size(); j++) {
			result[i] = a[i] - b[j];	
		}
	}
	return result;
}

