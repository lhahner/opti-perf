#include "benchmark/mnist.h"

void Mnist::load_batches()
	{
	torch::Device device(torch::kCPU);
	if(torch::cuda::is_available()){
		std::cout << "CUDA is available! Training on GPU." << std::endl;
		device = torch::Device(torch::kCUDA);
	} else {
		std::cout << "CUDA is not available! Training on GPU not possible." << std::endl;
	}
	DCGANGenerator generator(kNoiseSize);
	generator->to(device);

	nn::Sequential discriminator = create_discriminator();
	discriminator->to(device);

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
    			torch::Tensor real_images = batch.data.to(device);
    			torch::Tensor real_labels = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
    			torch::Tensor real_output = discriminator->forward(real_images).reshape(real_labels.sizes());
    			torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
    			d_loss_real.backward();

    			// Train discriminator with fake images.
    			torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1}, device);
    			torch::Tensor fake_images = generator->forward(noise);
    			torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
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
