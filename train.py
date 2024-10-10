import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CascadedVAE
from loss import cascaded_vae_loss as criterion
from invert import sample_images
from utils import save_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_channels = 3  # For RGB images (CIFAR-10 or similar)
latent_dims = [128, 64, 32]  # Three levels of latent spaces for the cascaded VAE

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = datasets.CIFAR10(root='cifar_data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Model and optimizer
model = CascadedVAE(input_channels, latent_dims).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 20000
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for i, (x, _) in enumerate(dataloader):  # Assuming a DataLoader is used
        x = x.to(device)  # No need to flatten as we're using convolutional layers
        optimizer.zero_grad()
        
        # Forward pass
        recon_list, mu_list, logvar_list = model(x)
        
        # Compute loss
        loss = criterion(recon_list, x, mu_list, logvar_list)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

    # Every visualize_every epochs, show generated images
    if (epoch + 1) % x == 0:
        sample_images_batch = sample_images(model, num_samples=8, image_size=32)
        # Assuming `generated_images` is a tensor of images
        save_images(sample_images_batch, num_images=8, save_dir="./output_images", grid_filename="generated_grid.png")
