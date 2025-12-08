#include <cstdint>
#include <string>
#include <torch/torch.h>
#include "dcgan.cpp"

class Mnist {
	public:
		const std::string DATASET_PATH = "./data/mnist/";
		const int64_t kBatchSize = 64;
		const int64_t kNumberOfEpochs = 10;
		const int64_t batches_per_epoch = 16;
		void load_batches();
};


void Mnist::load_batches()
	{
	DCGANGenerator generator(kNoiseSize);
	nn::Sequential discriminator = create_discriminator();

	auto dataset = torch::data::datasets::MNIST(DATASET_PATH)
			.map(torch::data::transforms::Normalize<>(0.5, 0.5))
			.map(torch::data::transforms::Stack<>());
	
	auto data_loader = torch::data::make_data_loader(std::move(dataset),
		        	torch::data::DataLoaderOptions()
				.batch_size(kBatchSize).workers(2));
	
	torch::optim::Adam generator_optimizer(
				generator->parameters(), 
				torch::optim::AdamOptions(2e-4)
				.betas(std::make_tuple(0.5, 0.5))
			);

	torch::optim::Adam discriminator_optimizer(
			discriminator->parameters(),
			torch::optim::AdamOptions(2e-4)
			.betas(std::make_tuple(0.5, 0.5))
		);
 	for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
		int64_t batch_index = 0;
		for (torch::data::Example<>& batch : *data_loader) {
			// Train discriminator with real images.
    			discriminator->zero_grad();
    			torch::Tensor real_images = batch.data;
    			torch::Tensor real_labels = torch::empty(batch.data.size(0)).uniform_(0.8, 1.0);
    			torch::Tensor real_output = discriminator->forward(real_images).reshape(real_labels.sizes());
    			torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
    			d_loss_real.backward();

    			// Train discriminator with fake images.
    			torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1});
    			torch::Tensor fake_images = generator->forward(noise);
    			torch::Tensor fake_labels = torch::zeros(batch.data.size(0));
    			torch::Tensor fake_output = discriminator->forward(fake_images.detach())
				.reshape(fake_labels.sizes());
    			torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
    			d_loss_fake.backward();

 	   		torch::Tensor d_loss = d_loss_real + d_loss_fake;
    			discriminator_optimizer.step();
	
    			// Train generator.
    			generator->zero_grad();
    			fake_labels.fill_(1);
    			fake_output = discriminator->forward(fake_images).reshape(fake_labels.sizes());
    			torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
    			g_loss.backward();
    			generator_optimizer.step();

    			std::printf(
        		"\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f",
        			epoch,
        			kNumberOfEpochs,
        			++batch_index,
        			batches_per_epoch,
        			d_loss.item<float>(),
        			g_loss.item<float>());
  		}
	}		
}
