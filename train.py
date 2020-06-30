from __future__ import print_function
import os
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from utils import progress_bar
from utils import accuracy
warnings.simplefilter('ignore')

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
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == "cifar10":
    n_classes = 10
    trainset = datasets.CIFAR10(root="~/data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root="~/data", train=False, download=True, transform=transform)
elif args.dataset == "cifar100":
    n_classes = 100
    trainset = datasets.CIFAR100(root="~/data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR100(root="~/data", train=True, download=True, transform=transform)
else:
    raise NotImplementedError

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

if args.resume:
    # Load checkpoint
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
        + str(args.seed))

    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print("==> Building model..")
    net = models.resnet.ResNet18(n_classes=n_classes)

if not os.path.isdir("results"):
    os.mkdir("results")

net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)


def train(epoch, update=True, topk=(1,)):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0.0
    accuracies = []

    for i, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        preds = torch.softmax(outputs, 1)

        p_y = np.random.uniform(0, 1, (inputs.size(0), n_classes))
        p_y = torch.Tensor(p_y).to(device)
        p_y = torch.softmax(p_y, 1)

        R = nn.KLDivLoss()(p_y.log(), preds)
        loss = criterion(outputs, targets) + args.gamma * R

        train_loss += loss.item()
        accuracies += accuracy(outputs, targets, topk=topk)

        if update:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        progress_bar(i, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%%'
                     % (train_loss/(i+1), np.mean(accuracies)))

        return (train_loss/i, accuracies)


def test(epoch, update=True, topk=(1,)):
    global best_acc
    net.eval()
    test_loss = 0
    accuracies = []
    for i, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        accuracies += accuracy(outputs, targets, topk=topk)

        acc = np.mean(accuracies)

        progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%%'
                     % (test_loss/(i+1), acc))
    if update:
        if acc > best_acc:
            checkpoint(acc, epoch)
            best_acc = acc

    return (test_loss/i, accuracies)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname) and not args.test:
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])

if not args.test:
    for epoch in range(start_epoch, args.epoch):
        train_loss, reg_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        adjust_learning_rate(optimizer, epoch)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                                test_acc])
else:
    for k in [1, 5]:
        test_loss, test_acc = test(1, update=False, topk=(k,))
        train_loss, reg_loss, train_acc = train(1, update=False, topk=(k,))
        print("Top{} Train Acc=".format(k, np.mean(train_acc)))
        print("Top{} Test Acc=".format(k, np.mean(test_acc)))

    print("train_loss=", train_loss)
    print("test_loss=", test_loss)
    print("diff=", test_loss - train_loss)
