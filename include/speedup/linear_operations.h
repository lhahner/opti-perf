#include <stdexcept>
#include <vector>
#include "speedup/operations.h"

class LinearOperations : public Operations {
 public:
    enum Operation {
        ADD,
        SUB,
        MULT,
        DIV,
        TRANS,
        INVERSE
    };
  std::vector<float> add(std::vector<float> a, std::vector<float> b);

  std::vector<float> sub(std::vector<float> a, std:: vector<float> b);
  void execute(long unsigned int thread_count, Operation operations, std::vector<float> a, std::vector<float> b);
};
