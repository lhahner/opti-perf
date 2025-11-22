#include <stdexcept>
#include <vector>
#include "operations.h"

using namespace std;

class LinearOperations : public Operations {
 public:
  	vector<float> add(vector<float> a, vector<float> b);
  	vector<float> sub(vector<float> a, vector<float> b);
};
