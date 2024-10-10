import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_channels):
        super(Encoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels//2, kernel_size=3, stride=1, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(latent_channels//2, latent_channels//2, kernel_size=3, stride=1, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(latent_channels//2, latent_channels//2, kernel_size=3, stride=1, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Conv2d(latent_channels//2, latent_channels, kernel_size=3, stride=2, padding=1),  # 4x4 -> 2x2
            nn.ReLU()
        )
        
        # Convolutional layers to generate mean and log variance (same spatial dimensions)
        self.conv_mu = nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1)
        self.conv_logvar = nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Apply the convolutional layers
        h = self.conv_layers(x)  # Output size: [B, latent_channels, 2, 2]

        # Generate mean and log variance using convolutional layers
        mu = self.conv_mu(h)     # Output size: [B, latent_channels, 2, 2]
        logvar = self.conv_logvar(h)  # Output size: [B, latent_channels, 2, 2]

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_channels, out_channels):
        super(Decoder, self).__init__()
        
        # Transposed convolutions for decoding (latent space -> image)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, latent_channels//2, kernel_size=3, stride=2, padding=1),  # 2x2 -> 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(latent_channels//2, latent_channels//2, kernel_size=3, stride=1, padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(latent_channels//2, latent_channels//2, kernel_size=3, stride=1, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(latent_channels//2, out_channels, kernel_size=3, stride=1, padding=1),  # 16x16 -> 32x32
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, z):
        # Decode the latent 4D tensor back to the original image dimensions
        return self.deconv_layers(z)

