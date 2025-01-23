# NeuralCompression

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

### Performance

Performance is evaluated on a reserved test set in each case. No hyper-parameter tuning is performed. 

| Model | Data | Reconstruction Accuracy |
| :---:   | :---: | :---: |
| AE | MNIST  | |
| VAE | MNIST | | 
| VQ-VAE | MNIST | |  
| AE | CIFAR-10  | |
| VAE | CIFAR-10 | | 
| VQ-VAE | CIFAR-10 | |  
| AE | RGZ  | |
| VAE | RGZ | | 
| VQ-VAE | RGZ | |  

---


