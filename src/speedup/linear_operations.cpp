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
	for (long unsigned int i = 0; i<a.size(); i++) {
		for (long unsigned int j = 0; j<b.size(); j++) {
			result[i] = a[i] + b[j];	
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

void LinearOperations::execute(long unsigned int thread_count, 
                                             Operation operation, 
                                             std::vector<float> a, 
                                             std::vector<float> b)
{
    std::vector<float>* result = new std::vector;
    std::thread* threads = new std::thread[thread_count];
    if (thread_count < 1) {
        throw std::runtime_error("Thread count must be larger than 1");
    }
    if (!operation) {
        throw std::runtime_error("Operation to be considered is NULL");
    }
    switch(operation){
        case ADD:
            // Creating new threads
            for (long unsigned int thread = 0; thread < thread_count; thread++) { 
               threads[thread] = std::thread(&LinearOperations::add, this, a, b);
            }
            // Joining all threads
            for (long unsigned int thread = 0; thread < thread_count; thread++) {
                threads[thread].join();
            }
            delete threads;
            break;
        case SUB:
            // Creating new threads
            for (long unsigned int thread = 0; thread < thread_count; thread++) { 
                threads[thread] = std::thread(&LinearOperations::sub, this, a, b);

            }
            // Joining all threads
            for (long unsigned int thread = 0; thread < thread_count; thread++) {
               threads[thread].join(); 
            }
            delete threads;
            break;
        default:
            throw std::runtime_error("No concrete action found for execution");
    }
}
