import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from torchvision.utils import save_image

import wandb
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from models import VanillaAE, VariationalAE, VQVAE


# --------------------------------------------------------------------------

class Compressor(pl.LightningModule):

    """lightning module to reproduce resnet18 baseline"""

    def __init__(self, model, n_chan, imsize, hidden, latent_dim, lr):

        super().__init__()
        
        if model=='VanillaAE':
            self.model = VanillaAE(n_chan, hidden, latent_dim)
            self.K = (imsize/4)*latent_dim
        elif model=='VariationalAE':
            self.model = VariationalAE(n_chan, hidden, latent_dim)
            self.K = int(hidden/2)
        elif model=='VQVAE':
            self.model = VQVAE(n_chan, hidden, latent_dim)
            self.K = 512

        self.lr = lr
        
        # just used to create model summary:
        self.example_input_array = torch.zeros(1, n_chan, imsize, imsize)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        
        x_train, _ = batch
        x_tilde = self.model(x_train)
        
        nll, loss, recon_loss = self._get_losses(batch)
        
        self.log(f'train/recon', recon_loss)
        self.log(f'train/nll', nll)
        self.log(f'train/loss', nll+loss)
        
        return nll+loss

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, batch_idx, mode='val')
        return

    def test_step(self, batch, batch_idx):
        self._evaluate(batch, batch_idx, mode='test')
        return

    def _evaluate(self, batch, batch_idx, mode):

        x_test, _ = batch
        if mode == 'val' and batch_idx == 0:
            self._log_generate_images(x_test, mode)
        if mode == 'test' and batch_idx == 0:
            self._log_generate_images(x_test, mode)
            bpd = self._bits_per_dim(batch)
            #self.log(f'{mode}/bits_per_dim', bpd)

        nll, loss, recon_loss = self._get_losses(batch)

        self.log(f'{mode}/recon', recon_loss)
        self.log(f'{mode}/nll', nll)
        self.log(f'{mode}/loss', nll+loss)
        
        return

    def _log_generate_images(self, inputs, mode):

        if mode=='test':
            # select first 15 images from test set
            inputs = inputs[:15]

        outputs = self.model(inputs)
            
        size = inputs.size()[-1]
        n = inputs.size()[0]

        nchan = inputs.size()[1]
        imsize= inputs.size()[2]
        
        comparison = torch.zeros(nchan, 2*imsize, 15*imsize)
        
        for i in range(0,n):
            step = i*imsize
            comparison[:,:imsize,step:step+imsize] = inputs[i]
            comparison[:,imsize:,step:step+imsize] = outputs[i]

        self.logger.log_image(key=f'{mode}/reconstructions', images=[comparison])

        if mode=='test':
            save_image(comparison, 'images/reconstructions.png',nrow=1)
            
        return
        
    def _get_losses(self, batch):

        x_test, _ = batch
        x_tilde = self.model(x_test)

        # MSE / reconstruction loss
        mse_loss = nn.MSELoss(reduction='none')
        recon_loss = mse_loss(torch.squeeze(x_tilde), torch.squeeze(x_test)).sum()/x_test.size(0)
        
        # Negative log-likelihood
        nll = -Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(x_test).sum()/x_test.size(0)
    
        # model specific loss terms: 0 for AE; D_KL for VAE; codebook + commitment for VQVAE
        loss = self.model.loss
    
        return nll, loss, recon_loss

    def _bits_per_dim(self, batch, model='VanillaAE'):
    
        x_test, _ = batch
        
        x_tilde = self.model(x_test)
        
        mse_loss = nn.MSELoss(reduction='none')
        recon_loss = mse_loss(torch.squeeze(x_tilde), torch.squeeze(x_test)).sum()/x_test.size(0)
        
        # sum over all and then take batch average
        nll = -Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(x_test).sum()/x_test.size(0)
        log_px = nll.item() - np.log(self.K)
        log_px += self.model.loss # kl_d for VAE; 0 for AE
        log_px /= np.log(2)

        n_pixel = x_test.size()[-3]*x_test.size()[-1]**2
        bpd_const = np.log2(np.e) / n_pixel
        bpd = ((np.log(self.K) * n_pixel - recon_loss) * bpd_const)
        print(log_px, bpd)
        
        return bpd

    def configure_optimizers(self):

        # should update this at some point to take optimizer from config file
        optimizer    = torch.optim.Adam(self.parameters(), lr=self.lr)

        # learning rate steps specified in https://arxiv.org/pdf/1912.02175.pdf (A.3.1)
        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,40], gamma=0.1)

        #return [optimizer], [lr_scheduler]
        return [optimizer]

# -----------------------------------------------------------------------------
# --------------------------------------------------------------------------
