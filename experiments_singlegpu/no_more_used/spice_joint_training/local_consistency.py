#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys
sys.path.insert(0, './')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from spice.config import Config
from spice.data.build_dataset import build_dataset
from spice.model.build_model_sim import build_model_sim
from spice.model.sim2sem import Sim2Sem
from spice.solver import make_lr_scheduler, make_optimizer
from spice.utils.miscellaneous import mkdir, save_config
import numpy as np
from spice.utils.evaluation import calculate_acc, calculate_nmi, calculate_ari
from spice.utils.load_model_weights import load_model_weights


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    "--config-file",
    default="./configs/stl10/eval.py",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument(
    "--embedding",
    default="./results/stl10/embedding/feas_moco_512_l2.npy",
    type=str,
)


def main():
    torch.set_printoptions(linewidth=150)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_file)

    if not torch.cuda.is_available():
        raise NotImplementedError("You need GPU!")
    print("Use GPU: {} for training".format(torch.cuda.current_device()))

    # create SPICE model
    # first get encoder from self-supervised model and load wieghts
    moco_model = moco.builder.MoCo(
        base_encoder=resnet18_cifar,
        dim=cfg.moco.moco_dim, K=cfg.moco.moco_k, m=cfg.moco.moco_m, T=cfg.moco.moco_t, mlp=cfg.moco.mlp)
    
    # print(moco_model)
    # remove from encoder avgpool and fc layer in order to extract features
    moco_encoder = torch.nn.Sequential(*(list(moco_model.encoder_q.children())[:-2]))
    # print(moco_encoder)

    # print(cfg.clustering_head[0][0])
    clustering_head = SemHeadMulti(multi_heads=cfg.clustering_head[0])

    model = SPICEModel(feature_model=moco_encoder, head=clustering_head)
    # print(model)
    
    model.cuda()

    print("Loading model's weights")
    # if it's the first time, load weights of the self-supervised model
    # when is resumed it is not necessary since they will be resumed for all spice model later
    loc = 'cuda:{}'.format(torch.cuda.current_device())
    checkpoint = torch.load(args.pretrained_self_supervised_folder, map_location=loc)
    model.load_state_dict(checkpoint['state_dict'])

    
    cudnn.benchmark = True

    # training dataset with non augmented images
    train_original_images = CIFAR10(root=args.dataset_folder, train=True, 
        transform=transforms.Compose([transforms.ToTensor(),
                                    CIFAR10_normalization]),
        download=True)

    train_original_images_loader = torch.utils.data.DataLoader(
        train_original_images, batch_size=cfg.target_sub_batch_size, shuffle=False, # don't shuffle, the order is important because features are calculated using the original dataset
        num_workers=1, pin_memory=True, drop_last=True)

    model.eval()

    num_heads = len(cfg.model.head.multi_heads)
    assert num_heads == 1
    gt_labels = []
    pred_labels = []
    scores_all = []
    features_all = []

    # extract features and clustering scores from images
    for _, (images, labels) in enumerate(train_original_images_loader):
        images = images.cuda(non_blocking=True)
        
        with torch.no_grad():
            features = model.extract_only_features(images)
            scores = model.sem(images)

        assert len(scores) == num_heads

        # use only the first head score because 
        # the best model was saved with the best head in first position
        
        pred_idx = scores[0].argmax(dim=1)
        pred_labels.append(pred_idx)
        scores_all.append(scores[0])
        features_all.append(features)
        gt_labels.append(labels)

    gt_labels = torch.cat(gt_labels).long().cpu().numpy()
    features_all = torch.cat(features_all, dim=0)
    pred_labels = torch.cat(pred_labels).long().cpu().numpy()
    scores = torch.cat(scores_all).cpu()

    try:
        acc = calculate_acc(pred_labels, gt_labels)
    except:
        acc = -1

    nmi = calculate_nmi(pred_labels, gt_labels)
    ari = calculate_ari(pred_labels, gt_labels)

    print("ACC: {}, NMI: {}, ARI: {}".format(acc, nmi, ari))

    # get reliable samples and labels (relative clusters)
    reliable_samples_indices, reliable_samples_labels = model.local_consistency(features_all, scores)

    gt_labels_select = gt_labels[reliable_samples_indices]

    acc = calculate_acc(reliable_samples_labels, gt_labels_select)
    print('ACC of local consistency: {}, number of samples: {}'.format(acc, len(gt_labels_select)))

    labels_correct = np.zeros([features_all.shape[0]]) - 100
    labels_correct[reliable_samples_indices] = reliable_samples_labels

    np.save("{}/labels_reliable_{:4f}_{}.npy".format(cfg.results.output_dir, acc, len(reliable_samples_indices)), labels_correct)


if __name__ == '__main__':
    main()
