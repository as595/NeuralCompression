# NeuralCompression
AE, VAE, VQVAE comparison for task-oriented compression

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

### Benchmarks

<ins>Performance</ins>

| Model | Data | Reconstruction Accuracy |
| :---:   | :---: | :---: |
| AE | MNIST  | |
| VAE | MNIST | | 
| VQ-VAE | MNIST | |  
