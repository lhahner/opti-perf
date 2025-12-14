#include <iostream>
#include "benchmark/mnist.h"

int main() {
    Mnist mnist;
    mnist.load_batches();
    return 0;
}
