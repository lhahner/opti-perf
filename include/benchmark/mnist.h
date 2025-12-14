#include <cstdint>
#include <string>
#include <torch/torch.h>
#include "benchmark/dcgan.h"

class Mnist {
	public:
		const std::string DATASET_PATH = "./data/mnist/";
		const int64_t kBatchSize = 64;
		const int64_t kNumberOfEpochs = 10;
		const int64_t batches_per_epoch = 16;
		void load_batches();
};
