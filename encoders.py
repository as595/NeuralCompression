import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from functools import reduce
from utils import pairwise, vq, vq_st


# --------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim, num_layers):
        super().__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        units = self._get_units()

        layers = []
        for in_units, out_units in pairwise(units):
            layers += [nn.Sequential(nn.Linear(in_units, out_units, bias=False),
                             nn.BatchNorm1d(out_units),
                             nn.ReLU(True))
                       ]
        layers += [nn.Linear(units[-1], self.latent_dim)]

        self.layers = nn.Sequential(*layers)
        

    def _get_units(self):
        in_units = reduce(lambda a, b: a * b, self.input_shape)
        shrinkage = int(pow(in_units // self.latent_dim, 1 / self.num_layers))
        units = [in_units // (shrinkage ** i) for i in range(self.num_layers)]

        return units

    def forward(self, x):

        x = torch.flatten(x, start_dim=1)
        x = self.layers(x)

        return x

# --------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, output_shape, latent_dim, num_layers):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.output_shape = output_shape

        units = self._get_units()

        layers = []
        for in_units, out_units in pairwise(units[:-1]):
            layers += [nn.Sequential(nn.Linear(in_units, out_units, bias=False),
                       nn.BatchNorm1d(out_units),
                       nn.ReLU(True))
                       ]
        layers += [nn.Sequential(nn.Linear(units[-2], units[-1]), nn.Tanh())]
        
        self.layers = nn.Sequential(*layers)

    def _get_units(self):
        final_units = reduce(lambda a, b: a * b, self.output_shape)
        shrinkage = int(pow(final_units // self.latent_dim, 1 / self.num_layers))
        units = [final_units // (shrinkage ** i) for i in range(self.num_layers)]
        units.reverse()
        units = [self.latent_dim] + units

        return units

    def forward(self, x):

        x = self.layers(x)
        x = x.view(-1, *self.output_shape)

        return x

# --------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)
        
# --------------------------------------------------------------------------
        
class OordEncoder(nn.Module):
    def __init__(self, n_chan, dim):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(n_chan, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            ResBlock(dim),
            ResBlock(dim)
        )
        
    def forward(self, x):
        return self.layers(x)
        
# --------------------------------------------------------------------------

class OordDecoder(nn.Module):
    def __init__(self, n_chan, dim):
        super().__init__()

        self.layers = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, n_chan, 4, 2, 1),
            nn.Tanh()
        )
  
    def forward(self, x):
        return self.layers(x)

# --------------------------------------------------------------------------

