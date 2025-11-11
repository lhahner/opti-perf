#include <cstdlib>
#include <iostream>
#include "speedup/linear_operations.h"


int main()
{
	LinearOperations lo;
	std::cout << "Result: " << lo.add(2, 3) << '\n';
}
