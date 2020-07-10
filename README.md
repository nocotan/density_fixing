This repo contains demo implementation of the density-fixing training code in PyTorch based on the following paper:
> Masanari Mimura, Ryohei Izawa. Density Fixing: Simple yet Effective Regularization Method based on the Class Prior
> https://arxiv.org/abs/2007.03899

# Training

The following table shows the mean test errors w/ and w/o density-fixing regularization.


| Model                                | Top 1 Error | Top 5 Error |
|--------------------------------------|-------------|-------------|
| ResNet-18                            | 12.72%      | 0.812%      |
| ResNet-18 + density fixing (gamma=1) | 12.23%      | 0.779%      |

![fig:supervised_cifar10](figs/supervised_cifar10.png "Figure. Test error evolution for the best baseline model (ResNet-18) and density-fixing.")


# Acknowledgement
The CIFAR-10 reimplementation of density-fixing is adapted from the [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) repository by [kuangliu](https://github.com/kuangliu).
