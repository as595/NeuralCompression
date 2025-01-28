# NeuralCompression - Under Construction

Evaluation of neural methods for task-oriented compression of radio astronomy data. 

This implementation uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [Weights and Biases](https://wandb.ai).

---

### Usage

To run: 

```bash
python main.py --config configs/baseline.cfg
```

Configuration files can be found in the [`configs` directory]().

<ins>Config Files</ins>

| Filename | Description | 
| :---:   | :---: |
| `baseline.cfg` | Vanilla Autoencoder (AE)  | 
| `vae.cfg` | Variational Autoencoder (VAE) |  
| `vqvae.cfg` | Vector Quantized Variational Autoencoder (VQ-VAE)  |  

---

### Models

All models have the encoder-decoder architecture defined in [van den Oord+ 2017](https://arxiv.org/pdf/1711.00937).

---

### Performance

Performance is evaluated on a reserved test set in each case. No hyper-parameter tuning is performed. 

<ins>MNIST</ins>

| Model | Data | NLL | bits/dim | Example Images (top: input; bottom: output) |
| :---:   | :---: | :---: | :---: | :---: |
| AE | MNIST  | 720.481 | | ![alt text](./images/ae_mnist.png) |
| VAE | MNIST | 739.032 | | ![alt text](./images/vae_mnist.png) |
| VQ-VAE | MNIST | 722.629 | | ![alt text](./images/vqvae_mnist.png) |

<ins>CIFAR-10</ins>

| Model | Data | NLL | bits/dim | Example Images (top: input; bottom: output) |
| :---:   | :---: | :---: | :---: | :---: |
| AE | CIFAR-10  | 2826.159 | | ![alt text](./images/ae_cifar.png) |
| VAE | CIFAR-10 | 2902.008 | | ![alt text](./images/vae_cifar.png) |
| VQ-VAE | CIFAR-10 | 2855.927 || ![alt text](./images/vqvae_cifar.png) |  

<ins>RGZ</ins>

| Model | Data | NLL | bits/dim | Example Images (top: input; bottom: output) |
| :---:   | :---: | :---: |:---: | :---: |
| AE | RGZ  | | | ![alt text](./images/ae_mnist.png) |
| VAE | RGZ | | ||
| VQ-VAE | RGZ | || |  

---


