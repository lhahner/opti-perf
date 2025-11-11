#include <stdexcept>
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
	float add(float a, float b);
	
	/**
	 * TODO
	 * Addition of n matrices computed on a CUDA-Device.
	 *
	 * @param The matrices you want to add together.
	 * @return the sum of the matrices.
	 */
	float sub(float a, float b);	

	/**
	 * TODO
	 * Addition of n matrices computed on a CUDA-Device.
	 *
	 * @param The matrices you want to add together.
	 * @return the sum of the matrices.
	 */
	float mul(float a, float b);
	
	/**
	 * TODO
	 * Addition of n matrices computed on a CUDA-Device.
	 *
	 * @param The matrices you want to add together.
	 * @return the sum of the matrices.
	 */
	float div(float a, float b);	
};

