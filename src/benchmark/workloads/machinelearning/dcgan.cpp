#include "../../../../include/benchmark/workloads/machinelearning/dcgan.h"

// The size of the noise vector fed to the generator.
const int64_t kNoiseSize = 100;

// Number of epochs
const int64_t kNumberOfEpochs = 30;

// The batch size for training.
const int64_t kBatchSize = 64;

// Where to find the MNIST dataset.
const char *kDataFolder = "./mnist";

// After how many batches to create a new checkpoint periodically.
const int64_t kCheckpointEvery = 900;

// How many images to sample at every checkpoint.
const int64_t kNumberOfSamplesPerCheckpoint = 10;

// Set to `true` to restore models and optimizers from previously saved
// checkpoints.
const bool kRestoreFromCheckpoint = false;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

DCGANGeneratorImpl::DCGANGeneratorImpl(int kNoiseSize)
  : conv1(nn::ConvTranspose2dOptions(kNoiseSize, 256, 4)
              .bias(false)),
    batch_norm1(256),
    conv2(nn::ConvTranspose2dOptions(256, 128, 3)
              .stride(2)
              .padding(1)
              .bias(false)),
    batch_norm2(128),
    conv3(nn::ConvTranspose2dOptions(128, 64, 4)
              .stride(2)
              .padding(1)
              .bias(false)),
    batch_norm3(64),
    conv4(nn::ConvTranspose2dOptions(64, 1, 4)
              .stride(2)
              .padding(1)
              .bias(false))
{
 // register_module() is needed if we want to use the parameters() method later on
 register_module("conv1", conv1);
 register_module("conv2", conv2);
 register_module("conv3", conv3);
 register_module("conv4", conv4);
 register_module("batch_norm1", batch_norm1);
 register_module("batch_norm2", batch_norm2);
 register_module("batch_norm3", batch_norm3);
}

torch::Tensor DCGANGeneratorImpl::forward(torch::Tensor x)
{
  x = torch::relu(batch_norm1(conv1(x)));
  x = torch::relu(batch_norm2(conv2(x)));
  x = torch::relu(batch_norm3(conv3(x)));
  x = torch::tanh(conv4(x));
  return x;
}

/**
 * The discriminator receives real images from the MNIST dataset, or fake images from the generator. 
 * It is asked to emit a probability judging how real (closer to 1) or fake (closer to 0) a particular image is. 
 * Feedback from the discriminator on how real the images produced by the generator are is used to train the generator. 
 * Feedback on how good of an eye for authenticity the discriminator has is used to optimize the discriminator. 
 * In theory, a delicate balance between the generator and discriminator makes them improve in tandem, 
 * leading to the generator producing images indistinguishable from the target distribution, 
 * fooling the discriminator’s (by then) excellent eye into emitting a probability of 0.5 for both real and fake images.
 **/
nn::Sequential create_discriminator()
{
    return nn::Sequential(
        // Layer 1
        nn::Conv2d(nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // Layer 2
        nn::Conv2d(nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(128),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // Layer 3
        nn::Conv2d(
            nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(256),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // Layer 4
        nn::Conv2d(nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
        nn::Sigmoid());
}
