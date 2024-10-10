import torch
import torch.nn as nn
import torch.nn.functional as F
from components import Encoder, Decoder

class CascadedVAE(nn.Module):
    def __init__(self, input_channels, latent_dims):
        """
        Args:
            input_channels (int): Number of input channels (e.g., 3 for RGB images).
            latent_dims (list): List of latent dimensions for each VAE in the cascade.
        """
        super(CascadedVAE, self).__init__()
        
        # Create the encoder and decoder stacks
        self.encoders = nn.ModuleList([Encoder(input_channels, latent_dims[0])])
        self.decoders = nn.ModuleList([Decoder(latent_dims[-1], input_channels)])

        # Stack more encoders and decoders
        for i in range(1, len(latent_dims)):
            self.encoders.append(Encoder(latent_dims[i-1], latent_dims[i]))
            self.decoders.insert(0, Decoder(latent_dims[i], latent_dims[i-1]))

        a = 1
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Generates a tensor of random noise with the same shape as `std`
        return mu + eps * std  # Element-wise reparameterization

    def forward(self, x):
        # Encoding
        mu_list, logvar_list, z_list = [], [], []
        
        for encoder in self.encoders:
            mu, logvar = encoder(x)
            z = self.reparameterize(mu, logvar)
            mu_list.append(mu)
            logvar_list.append(logvar)
            z_list.append(z)
            x = z  # Use the latent representation for the next encoder
        
        # Decoding
        recon = z_list[-1]  # Start decoding from the last latent space
        for i, decoder in enumerate(self.decoders):
            recon = decoder(recon)  # Decode step by step
        
        return recon, mu_list, logvar_list
