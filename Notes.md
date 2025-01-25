# Notes on 1711.00937

---

## Architecture

The architecture for the models is described in [arXiv:1711.00937](https://arxiv.org/pdf/1711.00937) as: 

*The encoder consists of 2 strided convolutional layers with stride 2 and window size 4 × 4, followed by two residual
3 × 3 blocks (implemented as ReLU, 3x3 conv, ReLU, 1x1 conv), all having 256 hidden units. The
decoder similarly has two residual 3 × 3 blocks, followed by two transposed convolutions with stride
2 and window size 4 × 4.* 
(Section 4.1)

And they also say:

*In our experiments we define N discrete latents (e.g., we use a field of 32 x 32 latents for ImageNet,
or 8 x 8 x 10 for CIFAR10)*
(Section 3.2)

This second statement means that the residual blocks must have *N* hidden units (not 256) unless there is an additional FC layer after the encoder and before the decoder. I've assumed 

 * that there's no additional FC layer;
 * that there's a batchnorm after each conv layer.

Implemented as:

```python
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

class OordEncoder(nn.Module):
    def __init__(self, n_chan, hidden, zdim):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(n_chan, hidden, 4, 2, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(True),
            nn.Conv2d(hidden, zdim, 4, 2, 1),
            nn.BatchNorm2d(zdim),
            ResBlock(zdim),
            ResBlock(zdim)
        )
        
    def forward(self, x):
        return self.layers(x)
```

and similar for the decoder. Where for CIFAR-10:
n_chan = 3
hidden = 256
z_dim  = 10

---

## Loss Functions

### Vanilla AE

The loss for training the Vanilla AE model is an MSE loss:

$$\mathcal{L} = \frac{1}{N_{\rm batch}} \sum_{i=1}^{N_{\rm batch}} \sum_{j=1}^{N_{\rm chan}} \sum_{k=1}^{N_{\rm pixels}} (\tilde{x}[i,j,k] - x[i,j,k])^2,
$$

where $x[i,j,k]$ is the tensor corresponding to the image input and $\tilde{x}[i,j,k]$ is the model output.

This is implemented as:

```python
import torch.nn as nn

criterion = nn.MSELoss(reduction='none')
loss = criterion(x_tilde, x_train).sum()/x_train.size(0)
```

### VQ-VAE

The loss for training the  VQ-VAE model is:

$$\mathcal{L} = L_{\rm recon} + L_{\rm codebook} + \beta \cdot L_{\rm commitment},
$$

**(1):**

$$L_{\rm recon} = \frac{1}{N_{\rm batch}} \sum_{i=1}^{N_{\rm batch}} \sum_{j=1}^{N_{\rm chan}} \sum_{k=1}^{N_{\rm pixels}} (\tilde{x}[i,j,k] - x[i,j,k])^2,
$$

implemented as:

```python
import torch.nn as nn

criterion = nn.MSELoss(reduction='none')
loss_recon = criterion(x_tilde, x_train).sum()/x_train.size(0)
```

**(2):**

$$L_{\rm codebook} = \frac{1}{N_{\rm batch}} \frac{1}{N_{\rm latent}} \sum_{i=1}^{N_{\rm batch}} \sum_{j=1}^{N_{\rm latent}} (z_q(x)[i,j] - sg[z_e(x)[i,j]])^2,
$$

implemented as:

```python
import torch.nn.functional as F

z_e_x = self.encoder(x)
z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
loss_cb = F.mse_loss(z_q_x, z_e_x.detach()) # default: reduction='mean'
```

**(2):**

$$L_{\rm commitment} = \frac{1}{N_{\rm batch}} \frac{1}{N_{\rm latent}} \sum_{i=1}^{N_{\rm batch}} \sum_{j=1}^{N_{\rm latent}} (sg[z_q(x)[i,j]] - z_e(x)[i,j])^2,
$$

implemented as:

```python
import torch.nn.functional as F

z_e_x = self.encoder(x)
z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
loss_commit = F.mse_loss(z_e_x, z_q_x.detach())  # default: reduction='mean'
```

The mean reduction follows from this statement in Section 3.2:

*The resulting loss L is identical, except that we get an average over N
terms for k-means and commitment loss – one for each latent.* [sic.]

I think this means that they average the loss over all the latents for the *codebook* and commitment loss. For CIFAR-10, there are $8\times 8\times 10$ latents, so $N_{\rm latent} = 640$.
