#include <cstdint>
#include <string>
#include <torch/torch.h>
#include "benchmark/dcgan.h"

class Mnist {
	public:
		const char* DATASET_PATH = "./data/mnist/";
		const int kBatchSize = 64;
		const int kNumberOfEpochs = 10;
		const int64_t kLogInterval = 10;
		const int64_t kCheckpointEvery = 900;
		const int64_t kNumberOfSamplesPerCheckpoint = 10;		
		void load_batches();
};
