import torch

def sample_images(model, num_samples=8, image_size=32, device='cuda'):
    model.eval()
    with torch.no_grad():
        # Start with the latent space of the deepest encoder in the cascade
        last_latent_dim = model.encoders[-1].fc_mu.out_features  # Latent dimension of the last encoder

        # Sample from a standard normal distribution in the latent space
        z = torch.randn(num_samples, last_latent_dim).to(device)

        # Decode through the decoders to generate images
        generated_images = z
        for decoder in model.decoders:
            generated_images = decoder(generated_images)

        # Clamp the values to be between [0, 1] for valid image visualization
        generated_images = torch.clamp(generated_images, 0, 1)
        
    return generated_images.cpu()
