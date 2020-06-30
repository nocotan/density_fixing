from __future__ import print_function
import argparse
import warnings
import torch
warnings.simplefilter('ignore')

def main():
    parser = argparse.ArgumentParser(description='Density-Fixing PyTorch Training')
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--gamma", type=float, default=0.1, help="density-fixing parameter")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--model", type=str, default="resnet18", help="model type (default: resnet18)")
    parser.add_argument("--name", type=str, default="0", help="name of run")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--n_epochs", type=int, default=200, help="total epochs to run")
    parser.add_argument("--decay", type=float, default=1e-4, help="weight decay")
    args = parser.parse_args()

    # check gpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # preparing data
    print("==> Preparing data...")