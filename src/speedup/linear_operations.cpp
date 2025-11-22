#include <stdexcept>
#include <pthread.h>
#include "speedup/linear_operations.h"
#include <vector>
#include <stdexcept>

using namespace std;

vector<float> LinearOperations::add(vector<float> a, vector<float> b)
{
    if ((a.size() == 0 || b.size()) == 0 || (a.size() != b.size())) {
        runtime_error("matrix null or not same size.");
    }
    vector<float> result(a.size());
    #pragma omp parallel for
    for (long unsigned int i = 0; i<a.size(); i++) {
	result[i] = a[i] + b[i];
    }
    return result;
}

vector<float> LinearOperations::sub(vector<float> a, vector<float> b)
{
    if ((a.size() == 0 || b.size()) == 0 || (a.size() != b.size())) {
        runtime_error("matrix null or not same size.");
    }
    vector<float> result(a.size());
    #pragma omp parallel for
    for (long unsigned int i = 0; i<a.size(); i++) {
	result[i] = a[i] - b[i];
    }
    return result;
}
