# TransGAN: Transformer-based GAN

A lightweight implementation of **TransGAN**, where both the generator and discriminator are built with transformer blocks. Designed for image generation tasks. The implementation uses the CIFAR10 dataset, which consists of 60,000 32x32 color images across 10 classes, to show how these techniques address common GAN training issues such as mode collapse and training instability.If you have any suggestions, please contact us.

##  Structure

- `TransGAN.ipynb`: Main training and architecture code
- `checkpoints/`: Saved model weights
- `samples/`: Generated images
- `logs/`: TensorBoard logs

##  Features

- Transformer-based GAN (no convolutions)
- Supports WGAN-GP, spectral norm, feature matching, etc.
- Configurable via `GANConfig` dataclass

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy, Matplotlib, SciPy, Seaborn, tqdm
