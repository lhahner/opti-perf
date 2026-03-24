#include "benchmark/workloads/mnist_linear/mnist_linear.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>

namespace
{
constexpr uint32_t kMnistImageMagic = 2051;
constexpr uint32_t kMnistLabelMagic = 2049;
}

MnistLinear::MnistLinear(const std::string &dataset_dir, int batch_size, int max_samples)
	: batch_size_(batch_size)
{
	if (batch_size_ <= 0)
	{
		throw std::invalid_argument("batch_size must be positive");
	}

	loadDataset(dataset_dir, max_samples);
	initializeParameters();
}

void MnistLinear::runForward()
{
	if (sample_count_ == 0)
	{
		throw std::logic_error("MNIST dataset is empty");
	}

	const int batch_index = step_ % ((sample_count_ + batch_size_ - 1) / batch_size_);
	loadBatch(batch_index);

	std::fill(logits_.begin(), logits_.end(), 0.0f);
	std::fill(probs_.begin(), probs_.end(), 0.0f);
	std::fill(grad_weights_.begin(), grad_weights_.end(), 0.0f);
	std::fill(grad_bias_.begin(), grad_bias_.end(), 0.0f);

	double loss_accumulator = 0.0;
	for (int sample = 0; sample < batch_size_; ++sample)
	{
		const float *x = batch_x_.data() + static_cast<size_t>(sample) * input_dim_;
		float *sample_logits = logits_.data() + static_cast<size_t>(sample) * num_classes_;
		float *sample_probs = probs_.data() + static_cast<size_t>(sample) * num_classes_;

		for (int cls = 0; cls < num_classes_; ++cls)
		{
			float sum = bias_[cls];
			const float *weight_row = weights_.data() + static_cast<size_t>(cls) * input_dim_;
			for (int feature = 0; feature < input_dim_; ++feature)
			{
				sum += weight_row[feature] * x[feature];
			}
			sample_logits[cls] = sum;
		}

		loss_accumulator += stableSoftmaxCrossEntropy(sample_logits, batch_y_[sample], num_classes_, sample_probs);

		for (int cls = 0; cls < num_classes_; ++cls)
		{
			const float error = sample_probs[cls] - (cls == batch_y_[sample] ? 1.0f : 0.0f);
			grad_bias_[cls] += error;
			float *grad_row = grad_weights_.data() + static_cast<size_t>(cls) * input_dim_;
			for (int feature = 0; feature < input_dim_; ++feature)
			{
				grad_row[feature] += error * x[feature];
			}
		}
	}

	const float scale = 1.0f / static_cast<float>(batch_size_);
	for (float &value : grad_weights_)
	{
		value *= scale;
	}
	for (float &value : grad_bias_)
	{
		value *= scale;
	}

	loss_ = static_cast<float>(loss_accumulator * static_cast<double>(scale));
	++step_;
}

std::pair<int, float> MnistLinear::computeLoss()
{
	return {step_, loss_};
}

std::vector<HostParamView> MnistLinear::parameters()
{
	HostParamView weights_view;
	weights_view.data = weights_.data();
	weights_view.grad = grad_weights_.data();
	weights_view.count = weights_.size();
	weights_view.name = "linear.weight";

	HostParamView bias_view;
	bias_view.data = bias_.data();
	bias_view.grad = grad_bias_.data();
	bias_view.count = bias_.size();
	bias_view.name = "linear.bias";

	return {weights_view, bias_view};
}

void MnistLinear::loadDataset(const std::string &dataset_dir, int max_samples)
{
	int image_count = 0;
	int rows = 0;
	int cols = 0;
	all_images_ = loadImages(dataset_dir + "/train-images-idx3-ubyte", image_count, rows, cols);

	int label_count = 0;
	all_labels_ = loadLabels(dataset_dir + "/train-labels-idx1-ubyte", label_count);

	int test_image_count = 0;
	int test_rows = 0;
	int test_cols = 0;
	test_images_ = loadImages(dataset_dir + "/t10k-images-idx3-ubyte", test_image_count, test_rows, test_cols);

	int test_label_count = 0;
	test_labels_ = loadLabels(dataset_dir + "/t10k-labels-idx1-ubyte", test_label_count);

	if (image_count != label_count)
	{
		throw std::runtime_error("MNIST image/label count mismatch");
	}
	if (test_image_count != test_label_count)
	{
		throw std::runtime_error("MNIST test image/label count mismatch");
	}

	sample_count_ = image_count;
	input_dim_ = rows * cols;
	if (test_rows * test_cols != input_dim_)
	{
		throw std::runtime_error("MNIST train/test dimensions mismatch");
	}

	if (max_samples > 0 && max_samples < sample_count_)
	{
		sample_count_ = max_samples;
		all_images_.resize(static_cast<size_t>(sample_count_) * input_dim_);
		all_labels_.resize(sample_count_);
	}

	if (sample_count_ < batch_size_)
	{
		throw std::runtime_error("batch_size exceeds available MNIST samples");
	}

	batch_x_.assign(static_cast<size_t>(batch_size_) * input_dim_, 0.0f);
	batch_y_.assign(batch_size_, 0);
	logits_.assign(static_cast<size_t>(batch_size_) * num_classes_, 0.0f);
	probs_.assign(static_cast<size_t>(batch_size_) * num_classes_, 0.0f);
}

std::vector<float> MnistLinear::loadImages(const std::string &path, int &count, int &rows, int &cols)
{
	std::ifstream stream(path, std::ios::binary);
	if (!stream)
	{
		throw std::runtime_error("Failed to open MNIST image file: " + path);
	}

	const uint32_t magic = readBigEndianUInt32(stream);
	if (magic != kMnistImageMagic)
	{
		throw std::runtime_error("Invalid MNIST image file: " + path);
	}

	count = static_cast<int>(readBigEndianUInt32(stream));
	rows = static_cast<int>(readBigEndianUInt32(stream));
	cols = static_cast<int>(readBigEndianUInt32(stream));

	std::vector<uint8_t> raw(static_cast<size_t>(count) * rows * cols);
	stream.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(raw.size()));
	if (!stream)
	{
		throw std::runtime_error("Failed to read MNIST image payload: " + path);
	}

	std::vector<float> images(raw.size(), 0.0f);
	for (size_t i = 0; i < raw.size(); ++i)
	{
		images[i] = static_cast<float>(raw[i]) / 255.0f;
	}
	return images;
}

std::vector<uint8_t> MnistLinear::loadLabels(const std::string &path, int &count)
{
	std::ifstream stream(path, std::ios::binary);
	if (!stream)
	{
		throw std::runtime_error("Failed to open MNIST label file: " + path);
	}

	const uint32_t magic = readBigEndianUInt32(stream);
	if (magic != kMnistLabelMagic)
	{
		throw std::runtime_error("Invalid MNIST label file: " + path);
	}

	count = static_cast<int>(readBigEndianUInt32(stream));
	std::vector<uint8_t> labels(count, 0);
	stream.read(reinterpret_cast<char *>(labels.data()), static_cast<std::streamsize>(labels.size()));
	if (!stream)
	{
		throw std::runtime_error("Failed to read MNIST label payload: " + path);
	}
	return labels;
}

uint32_t MnistLinear::readBigEndianUInt32(std::ifstream &stream)
{
	unsigned char bytes[4] = {0, 0, 0, 0};
	stream.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
	if (!stream)
	{
		throw std::runtime_error("Failed to read MNIST header");
	}

	return (static_cast<uint32_t>(bytes[0]) << 24) |
		   (static_cast<uint32_t>(bytes[1]) << 16) |
		   (static_cast<uint32_t>(bytes[2]) << 8) |
		   static_cast<uint32_t>(bytes[3]);
}

void MnistLinear::initializeParameters()
{
	weights_.assign(static_cast<size_t>(num_classes_) * input_dim_, 0.0f);
	bias_.assign(num_classes_, 0.0f);
	grad_weights_.assign(weights_.size(), 0.0f);
	grad_bias_.assign(bias_.size(), 0.0f);

	std::mt19937 rng(12345);
	std::normal_distribution<float> dist(0.0f, 0.01f);
	for (float &weight : weights_)
	{
		weight = dist(rng);
	}
}

void MnistLinear::loadBatch(int batch_index)
{
	const size_t batch_offset = static_cast<size_t>(batch_index) * batch_size_;
	for (int i = 0; i < batch_size_; ++i)
	{
		const size_t dataset_index = (batch_offset + static_cast<size_t>(i)) % static_cast<size_t>(sample_count_);
		const float *source = all_images_.data() + dataset_index * input_dim_;
		float *target = batch_x_.data() + static_cast<size_t>(i) * input_dim_;
		std::copy(source, source + input_dim_, target);
		batch_y_[i] = all_labels_[dataset_index];
	}
}

float MnistLinear::stableSoftmaxCrossEntropy(const float *logits, int label, int classes, float *probs_out)
{
	float max_logit = logits[0];
	for (int i = 1; i < classes; ++i)
	{
		max_logit = std::max(max_logit, logits[i]);
	}

	double exp_sum = 0.0;
	for (int i = 0; i < classes; ++i)
	{
		const float shifted = logits[i] - max_logit;
		const float exp_value = std::exp(shifted);
		probs_out[i] = exp_value;
		exp_sum += exp_value;
	}

	for (int i = 0; i < classes; ++i)
	{
		probs_out[i] = static_cast<float>(probs_out[i] / exp_sum);
	}

	const float probability = std::max(probs_out[label], 1e-12f);
	return -std::log(probability);
}

float MnistLinear::evaluateTestAccuracy() const
{
	if (test_labels_.empty())
	{
		throw std::logic_error("MNIST test set is empty");
	}

	int correct = 0;
	std::vector<float> logits(num_classes_, 0.0f);
	for (size_t sample = 0; sample < test_labels_.size(); ++sample)
	{
		const float *x = test_images_.data() + sample * input_dim_;
		for (int cls = 0; cls < num_classes_; ++cls)
		{
			float sum = bias_[cls];
			const float *weight_row = weights_.data() + static_cast<size_t>(cls) * input_dim_;
			for (int feature = 0; feature < input_dim_; ++feature)
			{
				sum += weight_row[feature] * x[feature];
			}
			logits[cls] = sum;
		}

		int predicted = 0;
		for (int cls = 1; cls < num_classes_; ++cls)
		{
			if (logits[cls] > logits[predicted])
			{
				predicted = cls;
			}
		}
		if (predicted == test_labels_[sample])
		{
			++correct;
		}
	}

	return static_cast<float>(correct) / static_cast<float>(test_labels_.size());
}

float MnistLinear::evaluateTestLoss() const
{
	if (test_labels_.empty())
	{
		throw std::logic_error("MNIST test set is empty");
	}

	double loss_accumulator = 0.0;
	std::vector<float> logits(num_classes_, 0.0f);
	std::vector<float> probs(num_classes_, 0.0f);
	for (size_t sample = 0; sample < test_labels_.size(); ++sample)
	{
		const float *x = test_images_.data() + sample * input_dim_;
		for (int cls = 0; cls < num_classes_; ++cls)
		{
			float sum = bias_[cls];
			const float *weight_row = weights_.data() + static_cast<size_t>(cls) * input_dim_;
			for (int feature = 0; feature < input_dim_; ++feature)
			{
				sum += weight_row[feature] * x[feature];
			}
			logits[cls] = sum;
		}
		loss_accumulator += stableSoftmaxCrossEntropy(logits.data(), test_labels_[sample], num_classes_, probs.data());
	}

	return static_cast<float>(loss_accumulator / static_cast<double>(test_labels_.size()));
}
