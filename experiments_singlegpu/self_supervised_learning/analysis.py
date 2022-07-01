#!/usr/bin/env python

import argparse
import math
import os
import shutil
import random
import sys
sys.path.insert(0, './')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from PIL import Image

from SPICE.spice.config import Config

import torchvision.models as models
from SPICE.spice.model.feature_modules.resnet_cifar import resnet18_cifar

import SPICE.moco.loader
import SPICE.moco.builder

import torchvision.transforms as transforms
from experiments_singlegpu.datasets.utils.custom_transforms import PadToSquare
from experiments_singlegpu.datasets.utils.custom_transforms import DoNothing
from torchvision.datasets import CIFAR10
from experiments_singlegpu.datasets.CIFAR10_custom import CIFAR10Pair
from experiments_singlegpu.datasets.SocialProfilePictures import SocialProfilePictures, SocialProfilePicturesPair

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from experiments_singlegpu.self_supervised_learning.utils import extract_features_targets

parser = argparse.ArgumentParser(description='PyTorch MoCo Training')
parser.add_argument('--dataset_folder', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('--save_folder', metavar='DIR', default='./results/checkpoints/',
                    help='path to save checkpoints')

def data_augmentation(args):
    dataset_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    # https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/data_aug/contrastive_learning_dataset.py#L8
    mocov2_augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([SPICE.moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        dataset_normalization
        ]
    dataset = SocialProfilePicturesPair(version=3, randomize_metadata=True, root=args.dataset_folder, split="val", 
                        transform=transforms.Compose(
                                    [PadToSquare(),   
                                    transforms.Resize([224, 224])] +
                                    mocov2_augmentation))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=1, 
        pin_memory=True, drop_last=True)
    
    randomlist = []
    for i in range(0,20):
        n = random.randint(1,len(loader))
        randomlist.append(n)
    transform=transforms.Compose([PadToSquare(),   
                                    transforms.Resize([224, 224])])
    size = 0
    for i, (img1, img2) in enumerate(loader):
        img1, img2 = img1.cuda(), img2.cuda(),
        if i in randomlist:
            print("file path: {}/{}".format(dataset.metadata[size]['img_folder'], dataset.metadata[size]['img_name']))
            fig, ax = plt.subplots()
            ax.imshow(img1[0].cpu().numpy().transpose([1, 2, 0]))
            plt.savefig(os.path.join(args.save_folder, "image_{}_aug_1.jpg".format(i)))
            plt.close()
            fig, ax = plt.subplots()
            ax.imshow(img2[0].cpu().numpy().transpose([1, 2, 0]))
            plt.savefig(os.path.join(args.save_folder, "image_{}_aug_2.jpg".format(i)))
            plt.close()
            fig, ax = plt.subplots()
            ax.imshow(img2[0].cpu().numpy().transpose([1, 2, 0]))
            plt.savefig(os.path.join(args.save_folder, "image_{}_aug_2.jpg".format(i)))
            plt.close()
            original = Image.open(os.path.join(args.dataset_folder, dataset.metadata[size]['img_folder'], dataset.metadata[size]['img_name']))
            original = transform(original)
            fig, ax = plt.subplots()
            ax.imshow(original)
            plt.savefig(os.path.join(args.save_folder, "image_{}_original.jpg".format(i)))
            plt.close()
        size += img1.size()[0]
        print("[{}]/[{}] batch iteration".format(i, len(loader)))
                                            
    
def main():
    args = parser.parse_args()
    data_augmentation(args)


if __name__ == '__main__':
    main()
