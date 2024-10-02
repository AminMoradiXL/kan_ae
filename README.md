# Kolmogorov-Arnold Network Autoencoders (KAN-AE)

This repository contains the implementation of **Kolmogorov-Arnold Network (KAN) Autoencoders**, which leverage the Kolmogorov-Arnold representation theorem for edge-based activation functions. Our approach is designed to explore and enhance the representation of image data using KANs [1] in place of traditional CNN layers in autoencoders.

We compare the performance of KAN-based autoencoders to traditional convolutional autoencoders across several popular datasets, including **MNIST**, **SVHN**, and **CIFAR-10**.

## Features
- Implementation of KAN-based autoencoders for image data.
- Comparisons between KAN Autoencoders and traditional Convolutional Autoencoders (CNN-AE).
- Evaluation on image datasets: MNIST, SVHN, CIFAR-10.
- Includes efficient KAN [2] layer implementations to reduce memory overhead and computational complexity.

## Cite Our Work

If you find this project helpful, please cite our work:

```{python}
@article{moradi2024KAN_autoencoders,
  title={Kolmogorov-Arnold Network Autoencoders},
  author={Mohammadamin Moradi and Shirin Panahi and Erik Bollt and Ying-Cheng Lai},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## References
1. Liu, Ziming, et al. "Kan: Kolmogorov-arnold networks." arXiv preprint arXiv:2404.19756 (2024).
2. https://github.com/Blealtan/efficient-kan
