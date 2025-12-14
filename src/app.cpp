#include <cstdlib>
#include <iostream>
#include "speedup/linear_operations.h"
#include "benchmark/mnist.h"
#include <vector>

int main()
{
	Mnist mnist;
	mnist.load_batches();
}
