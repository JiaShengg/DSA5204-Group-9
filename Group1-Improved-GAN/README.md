# Improved-gan
This repository contains implementation code for the paper "Improved Techniques for Training GANs" by Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen.

- Feature matching
- Minibatch discrimination
- Historical averaging
- Virtual batch normalization
- One-sided label smoothing

The implementation uses the CIFAR10 dataset, which consists of 60,000 32x32 color images across 10 classes, to show how these techniques address common GAN training issues such as mode collapse and training instability.