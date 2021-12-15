#!/usr/bin/env python
import argparse
import math
import os
import shutil
import time
import sys
sys.path.insert(0, './')

import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from spice.model.feature_modules.resnet_cifar import resnet18_cifar

import moco.loader
import moco.builder
from torchvision.datasets import CIFAR10
from experiments_singlegpu.CIFAR10.CIFAR10_custom import CIFAR10Pair

import matplotlib.pyplot as plt
import numpy as np
from experiments_singlegpu.torch_lr_finder.lr_finder import LRFinder


parser = argparse.ArgumentParser(description='PyTorch MoCo lr finder')
parser.add_argument('--dataset_folder', metavar='DIR', default='./datasets/cifar10',
                    help='path to dataset')
parser.add_argument('--save_folder', metavar='DIR', default='./results/cifar10/moco',
                    help='path to results')
parser.add_argument('--logs_folder', metavar='DIR', default='./results/cifar10/moco/logs',
                    help='path to tensorboard logs')
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--start_lr', '--starting-learning-rate', default=1e-7, type=float,
                    help='starting learning rate')
parser.add_argument('--end_lr', '--ending-learning-rate', default=0.1, type=float,
                    help='ending learning rate')
parser.add_argument('--num_iter', '--number-iterations', default=100, type=int,
                    help='number of steps between start_lr and end_lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')



# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# code from https://github.com/davidtvs/pytorch-lr-finder
def main():
    args = parser.parse_args()
    print("moco lr finder started with params")
    print(args)

    # Data loading code
    CIFAR10_normalization = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    # https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/data_aug/contrastive_learning_dataset.py#L8
    mocov2_augmentation = [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        CIFAR10_normalization
    ]
    
    # creating CIFAR10 train and test dataset from custom CIFAR10 class
    train_dataset = CIFAR10Pair(root=args.dataset_folder, train=True, 
        transform=transforms.Compose(mocov2_augmentation), 
        download=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, 
        pin_memory=True, drop_last=True)
    test_dataset = CIFAR10Pair(root=args.dataset_folder, train=False, 
        transform=transforms.Compose(mocov2_augmentation), 
        download=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, 
        pin_memory=True)
    
    model = moco.builder.MoCo(
        base_encoder=resnet18_cifar,
        dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t, mlp=args.mlp, input_size=32, single_gpu=True)
    # print(model)

    torch.cuda.set_device(torch.cuda.current_device())
    model = model.cuda()
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = torch.optim.SGD(model.parameters(),     
    #                             args.start_lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), args.start_lr, weight_decay=args.weight_decay)

    lr_finder = LRFinder(model, optimizer, criterion)
    lr_finder.range_test(train_loader, end_lr=args.end_lr, num_iter=args.num_iter, step_mode="exp", accumulation_steps=1)
    losses = lr_finder.history
    print(losses)
    lr_finder.save_plot(save_name="step_mode_exp__start_lr_" + str(args.start_lr) + "__end_lr_" + str(args.end_lr) + "__number_of_iterations_" + str(args.num_iter))
    lr_finder.reset()

    lr_finder.range_test(train_loader, end_lr=args.end_lr, num_iter=args.num_iter, step_mode="linear", accumulation_steps=1)
    losses = lr_finder.history
    print(losses)
    lr_finder.save_plot(save_name="step_mode_linear__start_lr_" + str(args.start_lr) + "__end_lr_" + str(args.end_lr) + "__number_of_iterations_" + str(args.num_iter))
    lr_finder.reset()
    

if __name__ == '__main__':
    main()