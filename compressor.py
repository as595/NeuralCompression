import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from models import VanillaAE, VariationalAE, VQVAE


# --------------------------------------------------------------------------

class Compressor(pl.LightningModule):

    """lightning module to reproduce resnet18 baseline"""

    def __init__(self, model, input_shape, latent_dim, num_layers, lr):

        super().__init__()
        
        if model=='VanillaAE':
            self.model = VanillaAE(input_shape, latent_dim, num_layers)
            self.criterion = nn.MSELoss(reduction='none')
        elif model=='VariationalAE':
            self.model = VariationalAE(input_shape, latent_dim, num_layers)
            self.criterion = nn.MSELoss(reduction='none')
        elif model=='VQVAE':
            self.model = VQVAE(input_shape, latent_dim)
            self.criterion = nn.MSELoss(reduction='none')

        self.lr = lr
        

    def training_step(self, batch, batch_idx):
        
        x_train, _ = batch
        x_tilde = self.model(x_train)
        
        loss = self.criterion(torch.squeeze(x_tilde), torch.squeeze(x_train)).mean(0).sum()
        loss += self.model.loss

        self.log("train_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):

        x_test, _ = batch
        x_tilde = self.model(x_test)

        loss = self.criterion(x_tilde, torch.squeeze(x_test)).mean()

        self.log("test_loss", loss)

        return

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

        loss, recon_loss = self._get_losses(batch)

        self.log(f'{mode}/recon', recon_loss)
        self.log(f'{mode}/loss', loss)
        
        return

    def _log_generate_images(self, inputs, mode):

        if mode=='test':
            # select first 15 images from test set
            inputs = inputs[:15,:,:]

        size = inputs.size()[-1]
        n = inputs.size()[0]

        outputs = self.model(inputs)
        comparison = torch.cat([torch.squeeze(inputs), torch.squeeze(outputs)], dim=1)

        temp = comparison[0,:,:]
        for i in range(1,n):
            temp = torch.cat([temp, torch.squeeze(comparison[i,:,:])], dim=1)

        self.logger.log_image(key=f'{mode}/reconstructions', images=[temp])

        return

    def _get_losses(self, batch):

        x_test, _ = batch
        x_tilde = self.model(x_test)

        loss = self.model.loss
        recon_loss = self.criterion(torch.squeeze(x_tilde), torch.squeeze(x_test)).mean(0).sum() 
        
        return loss, recon_loss


    def configure_optimizers(self):

        # should update this at some point to take optimizer from config file
        optimizer    = torch.optim.Adam(self.parameters(), lr=self.lr)

        # learning rate steps specified in https://arxiv.org/pdf/1912.02175.pdf (A.3.1)
        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,40], gamma=0.1)

        #return [optimizer], [lr_scheduler]
        return [optimizer]

# -----------------------------------------------------------------------------
# --------------------------------------------------------------------------