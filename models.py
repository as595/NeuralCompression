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
    def __init__(self, n_chan, latent_dim):
        super().__init__()

        self.encoder = OordEncoder(n_chan, latent_dim)
        self.decoder = OordDecoder(n_chan, latent_dim)

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        self.loss = self._loss()

        return x

    def _loss(self):
        return 0.0

# --------------------------------------------------------------------------

class VariationalAE(nn.Module):
    def __init__(self, n_chan, latent_dim):
        super().__init__()

        self.encoder = OordEncoder(n_chan, 2*latent_dim)
        self.decoder = OordDecoder(n_chan, latent_dim)


    def forward(self, x):

        mu, logvar = self.encoder(x).chunk(2, dim=1)
        self.loss = self._loss(mu,logvar) # KL divergence

        noise = torch.randn_like(mu)
        z = noise * logvar.mul(.5).exp() + mu  # reparameterisation trick

        x_tilde = self.decoder(z)

        return x_tilde

    def _loss(self, mu, logvar):

        q_z_x = Normal(mu, logvar.mul(.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        print(q_z_x.size(), p_z.size(), kl_divergence(q_z_x, p_z).size())
        stop
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()
        
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
    def __init__(self, input_shape, latent_dim, K=512, beta=0.25):
        super().__init__()

        self.encoder = OordEncoder(input_shape, latent_dim)
        self.decoder = OordDecoder(input_shape, latent_dim)
        self.codebook = VQEmbedding(K, latent_dim)
        
        self.beta = beta # coefficient for commitment loss
        
    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)

        self.loss = self._loss(z_q_x, z_e_x)
        
        return x_tilde

    def _loss(self, z_q_x, z_e_x):

        vq_loss = F.mse_loss(z_q_x, z_e_x.detach())
        commitment_loss = F.mse_loss(z_e_x, z_q_x.detach())
        
        return vq_loss + self.beta*commitment_loss

# --------------------------------------------------------------------------
