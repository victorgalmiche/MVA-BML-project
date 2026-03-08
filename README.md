# Bayesian for machine learning - Final project (Victor Galmiche, Jean Wallard)

## *Practical and Asymptotically Exact Conditional Sampling in Diffusion Models*

This project studies the article:

**Practical and Asymptotically Exact Conditional Sampling in Diffusion Models**  
Wu, Trippe, Naesseth, Blei, Cunningham  
NeurIPS 2023  
https://arxiv.org/abs/2306.17775

The objective of this project is to study the proposed method and reimplement the **Twisted Diffusion Sampling (TDS)** algorithm.

After a thorough reimplementation of the TDS mechanism, we apply it to several **image conditioning tasks** on the **MNIST dataset**.

## Dataset

We use the **MNIST dataset** of handwritten digits:

LeCun, Y., Cortes, C., & Burges, C.  
*MNIST handwritten digit database*  
http://yann.lecun.com/exdb/mnist/

## Tasks

The reimplemented method is evaluated on three conditioning tasks:

- **Inpainting**
- **Denoising**
- **Class-conditional generation** (generating samples of images corresponding to targeted digits)

## Reference

Wu, L., Trippe, B., Naesseth, C., Blei, D., Cunningham, J.  
*Practical and Asymptotically Exact Conditional Sampling in Diffusion Models*  
NeurIPS 2023  
https://arxiv.org/abs/2306.17775