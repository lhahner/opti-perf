#include <stdexcept>
#include <vector>
#include "speedup/operations.h"

class LinearOperations : public Operations {
 public:
  std::vector<float> add(std::vector<float> a, std::vector<float> b);

  std::vector<float> sub(std::vector<float> a, std:: vector<float> b);
};
