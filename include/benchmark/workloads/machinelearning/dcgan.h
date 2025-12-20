#include <cstdint>
#include <torch/torch.h>

namespace nn = torch::nn;

extern const int64_t kNoiseSize;

/**
 * The generator receives samples from a noise distribution,
 * and its aim is to transform each noise sample into an image
 * that resembles those of a target distribution, in our case
 * the MNIST dataset.
 **/
class DCGANGeneratorImpl : public nn::Module
{
 public:
	DCGANGeneratorImpl();
	DCGANGeneratorImpl(int kNoiseSize);
	torch::Tensor forward(torch::Tensor x);

 private:
	nn::ConvTranspose2d conv1, conv2, conv3, conv4;
    nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};

TORCH_MODULE(DCGANGenerator);
nn::Sequential create_discriminator();

