#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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


parser = argparse.ArgumentParser(description='PyTorch MoCo Training')
parser.add_argument('--dataset_folder', metavar='DIR', default='./datasets/cifar10',
                    help='path to dataset')
parser.add_argument('--save_folder', metavar='DIR', default='./results/cifar10/moco',
                    help='path to results')
parser.add_argument('--logs_folder', metavar='DIR', default='./results/cifar10/moco/logs',
                    help='path to tensorboard logs')
parser.add_argument('--run_id', default='exp1',
                    help='id for creating tensorboard folder')
parser.add_argument('--save-freq', default=1, type=int, metavar='N',
                    help='epoch frequency of saving model')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.015, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='./results/cifar10/moco/checkpoint_last.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


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


def main():
    args = parser.parse_args()
    print(args)

    # setting of save_folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # checking GPU and showing infos
    if not torch.cuda.is_available():
        raise NotImplementedError("You need GPU!")
    print("Use GPU: {} for training".format(torch.cuda.current_device()))

    main_worker(args)


def main_worker(args):
    # creating model MoCo using resnet18_cifar which is an implementation adapted for CIFAR10
    model = moco.builder.MoCo(
        base_encoder=resnet18_cifar,
        dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t, mlp=args.mlp, input_size=32, single_gpu=True)
    # print(model)

    torch.cuda.set_device(torch.cuda.current_device())
    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(os.path.join(args.save_folder, "checkpoint_last.pth.tar")):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(torch.cuda.current_device())
            checkpoint = torch.load(args.resume, map_location=loc)

            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            print("=> last metrics where: '{}'".format(checkpoint['metrics']))
            # print("Resume's state_dict:")
            # for param_tensor in model.state_dict():
            #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            #     print(param_tensor, "\t", model.state_dict()[param_tensor])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True  # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

    train_data = CIFAR10(root=args.dataset_folder, train=True, 
        transform=transforms.Compose([transforms.ToTensor()]),  
        download=True)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False, num_workers=1, 
        pin_memory=True)

    test_data = CIFAR10(root=args.dataset_folder, train=False, 
        transform=transforms.Compose([transforms.ToTensor()]), 
        download=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=1, 
        pin_memory=True)

    

    # tensorboard plotter
    train_writer = SummaryWriter(args.logs_folder + "/train_projector")
    test_writer = SummaryWriter(args.logs_folder + "/test_projector")

    create_projector(model.encoder_q, train_loader, test_loader, train_writer, test_writer, args)

    
def create_projector(model, train_loader, test_loader, train_writer, test_writer, args):
    model.eval()
    train_feature_bank, train_images_bank, test_feature_bank, test_images_bank =  [], [], [], []
    with torch.no_grad():
        # generate feature bank from train dataset
        for data, target in train_loader:
            # print(data.size())
            feature = model(data.cuda(non_blocking=True)) # for every sample in the batch let predict features NxC tensor
            feature = F.normalize(feature, dim=1)
            train_feature_bank.append(feature)    # create list of features [tensor1 (NxC), tensor2 (NxC), tensorM (NxC)] where M is the number of minibatches
            train_images_bank.append(data)

        train_feature_bank = torch.cat(train_feature_bank, dim=0).contiguous()  # concatenates all features tensors [NxC],[NxC],... to obtain a unique tensor of features 
                                                                    # for all the dataset DxC
        train_images_bank = torch.cat(train_images_bank, dim=0).contiguous()  # same for images
                                                                        
        for data, target in test_loader:
            feature = model(data.cuda(non_blocking=True)) # for every sample in the batch let predict features NxC tensor
            feature = F.normalize(feature, dim=1)
            test_feature_bank.append(feature)    # create list of features [tensor1 (NxC), tensor2 (NxC), tensorM (NxC)] where M is the number of minibatches
            test_images_bank.append(data)

        test_feature_bank = torch.cat(test_feature_bank, dim=0).contiguous()  # concatenates all features tensors [NxC],[NxC],... to obtain a unique tensor of features 
                                                                    # for all the dataset DxC
        test_images_bank = torch.cat(test_images_bank, dim=0).contiguous()  # same for images
    
    print("creating projectors")
    train_writer.add_embedding(train_feature_bank, metadata=train_loader.dataset.targets, label_img=train_images_bank) 
    test_writer.add_embedding(test_feature_bank, metadata=test_loader.dataset.targets, label_img=test_images_bank)    


if __name__ == '__main__':
    main()
