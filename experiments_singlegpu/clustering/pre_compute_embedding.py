#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import sys
sys.path.insert(0, './')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np

from SPICE.spice.model.feature_modules.resnet_cifar import resnet18_cifar
import SPICE.moco.builder
import SPICE.moco.loader
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='Features extraction from MoCo')
parser.add_argument('--dataset_folder', metavar='DIR', default='./datasets/cifar10',
                    help='path to dataset')
parser.add_argument('--model_path', type=str, default='results/128_0.5_200_512_500_model.pth',
                    help='The pretrained model path')
parser.add_argument('--save_folder', metavar='DIR', default='./results/cifar10/moco',
                    help='path to results')
parser.add_argument('--logs_folder', metavar='DIR', default='./results/cifar10/moco/logs',
                    help='path to tensorboard logs')        
                           
parser.add_argument('--batch-size', type=int, default=512, help='Number of images in each mini-batch')

# MoCo parameters, use the same of the saved model
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')


# new model to wrap encoder and load weights
class Net(nn.Module):
    def __init__(self, moco_model, pretrained_path):
        super(Net, self).__init__()

        # take query encoder of moco
        self.encoder_q = moco_model.encoder_q

        for param_tensor in self.encoder_q.state_dict():
            print(param_tensor, "\t", self.encoder_q.state_dict()[param_tensor].size())
        print(self.encoder_q.state_dict()['layer4.1.bn2.weight'])

        # load trained parameters
        loc = 'cuda:{}'.format(torch.cuda.current_device())
        checkpoint = torch.load(pretrained_path, map_location=loc)
        self.load_state_dict(checkpoint['state_dict'], strict=False)
        print(self.encoder_q.state_dict()['layer4.1.bn2.weight'])

        # remove the fc and avgpool layer from encoder_q in order to get features
        self.encoder_q = torch.nn.Sequential(*(list(self.encoder_q.children())[:-2]))
        # print(self.encoder_q.state_dict()['5.1.bn2.weight'])

    def forward(self, x):
        out = self.encoder_q(x)

        return out


def main():  
    args = parser.parse_args()
    print(args)

    # Data loading code
    CIFAR10_normalization = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    
    train_data = CIFAR10(root=args.dataset_folder, train=True, 
                        transform=transforms.Compose([transforms.ToTensor(),CIFAR10_normalization]), download=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    # test_data = CIFAR10(root=args.dataset_folder, train=False, 
    #                     transform=transforms.Compose([transforms.ToTensor(),CIFAR10_normalization]), download=True)
    # test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    moco_model = SPICE.moco.builder.MoCo(
        base_encoder=resnet18_cifar,
        dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t, mlp=args.mlp, input_size=32, single_gpu=True)

    # create new model with only query encoder
    model = Net( moco_model=moco_model, pretrained_path=args.model_path).cuda()

    print(model)

    cudnn.benchmark = True

    model.eval()

    pool = nn.AdaptiveAvgPool2d(1)

    feature_bank = []

    for i, (data, target) in enumerate(train_loader):
        data = data.cuda(non_blocking=True)
        # print(data.shape)
        with torch.no_grad():
            feature = model(data)
            # print(feature.shape)
            if len(feature.shape) == 4:
                # make avg reducing the 3rd and 4th dimension to 1
                feature = pool(feature)
                # make two dimensional vector
                feature = torch.flatten(feature, start_dim=1)
            # print(feature.shape)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature.cpu())  # use cpu in order to convert to numpy later

        print('Features extraction: [{}/{}]'.format( i, len(train_loader)))

    # print(feature_bank.shape)
    feature_bank = torch.cat(feature_bank, dim=0)
    feature_bank = feature_bank.numpy()

    np.save("{}/moco_features.npy".format(args.save_folder), feature_bank)


if __name__ == '__main__':
    main()