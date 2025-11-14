#include <vector>

class Operations {
public:
	virtual ~Operations() = default;
	virtual std::vector<float> add(std::vector<float> a, 
				       std::vector<float> b) = 0;
};
