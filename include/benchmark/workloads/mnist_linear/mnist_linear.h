#ifndef INCLUDE_BENCHMARK_WORKLOADS_MNIST_LINEAR_MNIST_LINEAR_H_
#define INCLUDE_BENCHMARK_WORKLOADS_MNIST_LINEAR_MNIST_LINEAR_H_

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/workloads/workload.h"

class MnistLinear : public Workload
{
public:
	explicit MnistLinear(const std::string &dataset_dir, int batch_size, int max_samples = 0);

	void runForward() override;
	std::pair<int, float> computeLoss() override;
	std::vector<HostParamView> parameters() override;
	float evaluateTestAccuracy() const;
	float evaluateTestLoss() const;

	const char *workloadType = "Training";
	const char *workloadName = "MNIST Linear Classifier";

	int batchSize() const { return batch_size_; }
	long inputSize() const { return static_cast<long>(input_dim_) * static_cast<long>(num_classes_); }

private:
	void loadDataset(const std::string &dataset_dir, int max_samples);
	static std::vector<float> loadImages(const std::string &path, int &count, int &rows, int &cols);
	static std::vector<uint8_t> loadLabels(const std::string &path, int &count);
	static uint32_t readBigEndianUInt32(std::ifstream &stream);
	void initializeParameters();
	void loadBatch(int batch_index);
	static float stableSoftmaxCrossEntropy(const float *logits, int label, int classes, float *probs_out);

	int input_dim_ = 0;
	int num_classes_ = 10;
	int batch_size_ = 0;
	int sample_count_ = 0;
	int step_ = 0;
	float loss_ = 0.0f;

	std::vector<float> all_images_;
	std::vector<uint8_t> all_labels_;
	std::vector<float> test_images_;
	std::vector<uint8_t> test_labels_;

	std::vector<float> batch_x_;
	std::vector<uint8_t> batch_y_;
	std::vector<float> logits_;
	std::vector<float> probs_;

	std::vector<float> weights_;
	std::vector<float> bias_;
	std::vector<float> grad_weights_;
	std::vector<float> grad_bias_;
};

#endif
