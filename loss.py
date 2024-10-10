import torch
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (assumes inputs are normalized between 0 and 1)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

def cascaded_vae_loss(recon_list, x, mu_list, logvar_list):
    """
    Computes the combined loss for the entire cascaded VAE.
    Args:
        recon_list: List of reconstructions from each VAE level.
        x: Original input data.
        mu_list: List of means (mu) from each VAE level.
        logvar_list: List of log variances (logvar) from each VAE level.
    """
    loss = 0.0
    
    # Loss for the first VAE (reconstructing the input)
    loss += vae_loss(recon_list[0], x, mu_list[0], logvar_list[0])
    
    # Loss for each subsequent VAE (reconstructing the latent space from the previous VAE)
    for i in range(1, len(recon_list)):
        loss += vae_loss(recon_list[i], mu_list[i-1], mu_list[i], logvar_list[i])
    
    return loss
