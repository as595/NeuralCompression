# https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/modules.py
# https://github.com/google-deepmind/sonnet/blob/v1/sonnet/python/modules/nets/vqvae.py
# https://nbviewer.org/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from utils import pairwise, vq, vq_st
from encoders import Encoder, Decoder, OordEncoder, OordDecoder

# --------------------------------------------------------------------------

class VanillaAE(nn.Module):

    """
    Vanilla Autoencoder 
    Architecture follows: https://arxiv.org/pdf/1711.00937 (Sec 4.1)
    
    Parameters:
        n_chan : number of inout channels
        hidden : number of hidden units
        latent_dim  : dimension of latent embedding space 
    """

    def __init__(self, n_chan, hidden, latent_dim):
        super().__init__()

        self.encoder = OordEncoder(n_chan, hidden, hidden)
        self.to_latent = nn.Conv2d(hidden, latent_dim, 1, 1, 1)
        self.from_latent = nn.ConvTranspose2d(latent_dim, hidden, 3, 1, 1)
        self.decoder = OordDecoder(n_chan, hidden, hidden)

    def forward(self, x):

        x = self.encoder(x)
        x = self.to_latent(x)
        x = self.from_latent(x)
        x = self.decoder(x)

        self.loss = self._loss()

        return x

    def _loss(self):
        return 0.0

# --------------------------------------------------------------------------

class VariationalAE(nn.Module):
    def __init__(self, n_chan, hidden, latent_dim, beta=1):
        super().__init__()

        self.encoder = OordEncoder(n_chan, hidden, hidden)
        self.to_latent = nn.Conv2d(hidden, 2*latent_dim, 1, 1, 1)
        self.from_latent = nn.ConvTranspose2d(latent_dim, hidden, 3, 1, 1)
        self.decoder = OordDecoder(n_chan, hidden, hidden)
        self.beta = beta

    def forward(self, x):

        x = self.encoder(x)
        mu, logvar = self.to_latent(x).chunk(2, dim=1)
        self.loss = self.beta*self._loss(mu,logvar) # KL divergence

        noise = torch.randn_like(mu)
        z = noise * logvar.mul(.5).exp() + mu  # reparameterisation trick

        z = self.from_latent(z)
        x_tilde = self.decoder(z)

        return x_tilde

    def _loss(self, mu, logvar):

        q_z_x = Normal(mu, logvar.mul(.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum()/mu.size(0)
        
        # https://arxiv.org/pdf/1312.6114 Appendix B
        # verified equivalent
        # d_kl = -1.*(0.5*(1 + logvar - mu.pow(2) - logvar.exp())).sum()/mu.size(0)
        
        return kl_div

# --------------------------------------------------------------------------

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar

# --------------------------------------------------------------------------

class VQVAE(nn.Module):


    """
    VQ-VAE
    Architecture follows: https://arxiv.org/pdf/1711.00937 (Sec 4.1)
    
    Parameters:
        n_chan : number of inout channels
        hidden : number of hidden units
        latent_dim  : dimension of latent embedding space 
        K      : dimension of codebook
        beta   : coefficient for commitment loss
    """
    
    def __init__(self, n_chan, hidden, latent_dim, K=512, beta=0.25):
        super().__init__()

        self.encoder = OordEncoder(n_chan, hidden, hidden)
        self.to_latent = nn.Conv2d(hidden, latent_dim, 1, 1, 1)
        self.codebook = VQEmbedding(K, latent_dim)
        self.from_latent = nn.ConvTranspose2d(latent_dim, hidden, 3, 1, 1)
        self.decoder = OordDecoder(n_chan, hidden, hidden)
        
        
        self.beta = beta # coefficient for commitment loss
        

    def forward(self, x):
        
        z_e_x = self.encoder(x)
        z_e_x = self.to_latent(z_e_x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        z_q_x_st = self.from_latent(z_q_x_st)
        x_tilde = self.decoder(z_q_x_st)

        self.loss = self._loss(z_q_x, z_e_x)
        
        return x_tilde

    def _loss(self, z_q_x, z_e_x):

        loss_cb = F.mse_loss(z_q_x, z_e_x.detach())
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
        
        return loss_cb + self.beta*loss_commit

# --------------------------------------------------------------------------
