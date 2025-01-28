# Notes on 1711.00937

---

## Architecture

The architecture for the models is described in [arXiv:1711.00937](https://arxiv.org/pdf/1711.00937) as: 

*The encoder consists of 2 strided convolutional layers with stride 2 and window size 4 × 4, followed by two residual
3 × 3 blocks (implemented as ReLU, 3x3 conv, ReLU, 1x1 conv), all having 256 hidden units. The
decoder similarly has two residual 3 × 3 blocks, followed by two transposed convolutions with stride
2 and window size 4 × 4.* 
(Section 4.1)

This suggests that *all* of the layers have 256 hidden units; however, they also say:

*In our experiments we define N discrete latents (e.g., we use a field of 32 x 32 latents for ImageNet,
or 8 x 8 x 10 for CIFAR10)*
(Section 3.2)

This second statement means that the residual blocks must have *N* hidden units (not 256), **unless** there is an additional layer after the encoder and before the decoder. So I've put in an extra layer to go from the encoder to the latent space, and an extra layer to go from the latent space to the decoder. I've also assumed that there's a batchnorm after each conv layer.


---

## Loss Functions

### Vanilla AE

The loss for training the Vanilla AE model is the NLL, $- \log p_{\theta} (\mathbf{x} | \mathbf{z})$:

$$\mathcal{L} = \frac{1}{N_{\rm batch}} \sum_{i=1}^{N_{\rm batch}} \sum_{j=1}^{N_{\rm chan}} \sum_{k=1}^{N_{\rm pixels}} 0.5\times \log(2\pi) - 0.5\times (\tilde{x}[i,j,k] - x[i,j,k])^2,
$$

where $x[i,j,k]$ is the tensor corresponding to the image input and $\tilde{x}[i,j,k]$ is the model output.

This is implemented as:

```python
from torch.distributions.normal import Normal

nll = -Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(x).sum()/x.size(0)
```

Note: this is the correct NLL for an image, which (assuming all pixels are independent) should be summed over the image rather than averaged. However, for astronomy images, pixels are not independent... try: `https://pytorch.org/docs/stable/distributions.html#multivariatenormal`

### VAE

The loss for training the Variational AE model is:

$$\mathcal{L} = \frac{1}{N_{\rm batch}} \sum_{i=1}^{N_{\rm batch}} D_{\rm KL} (q_{\phi}(\mathbf{z}|\mathbf{x}) || p_{\theta}(\mathbf{z}) ) - \log p_{\theta} (\mathbf{x} | \mathbf{z}),
$$

where $- \log p_{\theta} (\mathbf{x} | \mathbf{z})$ is the negative log-likelihood, and $D_{\rm KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) || p_{\theta}(\mathbf{z}) )$ is the KL-divergence.

This is implemented as:

```python
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

nll = -Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(x).sum()/x.size(0)

mu, logvar = self.encoder(x).chunk(2, dim=1)
q_z_x = Normal(mu, logvar.mul(.5).exp())
p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
kl_div = kl_divergence(q_z_x, p_z).sum()/mu.size(0)

```

### VQ-VAE

The loss for training the VQ-VAE model is:

$$\mathcal{L} = L_{\rm recon} + L_{\rm codebook} + \beta \cdot L_{\rm commitment},
$$

and from the original implementation it seems that all terms are taken as the average over their respective dimensions, i.e.

**(1):**

$$L_{\rm recon} = \frac{1}{N_{\rm batch}N_{\rm chan}N_{\rm pixels}} \sum_{i=1}^{N_{\rm batch}} \sum_{j=1}^{N_{\rm chan}} \sum_{k=1}^{N_{\rm pixels}} (\tilde{x}[i,j,k] - x[i,j,k])^2,
$$

implemented as:

```python
from torch.distributions.normal import Normal

nll = -Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(x).sum()/x.size(0)
loss = nll*x.size(0)/x.numel()
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

I think this means that they average the loss over all the latents for the *codebook* and commitment loss. For CIFAR-10, there are $8\times 8\times 10$ latents, so $N_{\rm latent} = 640$. This is also consistent with the tensorflow code in the [author's implementation](https://github.com/google-deepmind/sonnet/blob/v1/sonnet/python/modules/nets/vqvae.py).
