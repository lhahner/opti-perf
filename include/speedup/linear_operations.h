#include <stdexcept>
#include <vector>
#include "speedup/operations.h"

class LinearOperations : public Operations {
 public:
 /**
  * TODO
  * Addition of n matrices computed on a CUDA-Device.
  *
  * @param The matrices you want to add together.
  * @return the sum of the matrices.
  */
  std::vector<float> add(std::vector<float> a, std::vector<float> b);
};

