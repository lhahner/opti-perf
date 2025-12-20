#include <iostream>

#include "benchmark/workloads/machinelearning/mnist.h"

int main() {
    Mnist mnist;
    mnist.load_batches();
    return 0;
}
