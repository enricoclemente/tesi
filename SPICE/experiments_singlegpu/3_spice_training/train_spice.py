#!/usr/bin/env python
import argparse
import math
import os
import random
import shutil
import time
import sys
sys.path.insert(0, './')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

from spice.config import Config
from spice.data.augment import Augment, Cutout
from torchvision.datasets import CIFAR10
from experiments_singlegpu.datasets.CIFAR10.CIFAR10_custom import CIFAR10Pair

import moco.builder
from spice.model.feature_modules.resnet_cifar import resnet18_cifar
from experiments_singlegpu.spice_model import SPICEModel
from spice.model.heads.sem_head_multi import SemHeadMulti
from spice.utils.evaluation import calculate_acc, calculate_nmi, calculate_ari

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='SPICE training')


parser.add_argument("--config_file", default="./configs/stl10/spice_self.py", metavar="FILE",
                    help="path to config file", type=str)

# arguments for saving and resuming
parser.add_argument('--dataset_folder', metavar='DIR', default='./datasets/cifar10',
                    help='path to dataset')
parser.add_argument('--pretrained_self_supervised_folder', default='./results/cifar10/moco/checkpoint_last.pth.tar', type=str, metavar='PATH',
                    help='path to self-supervised checkpoint to take its weigths')
parser.add_argument('--resume', default='./results/cifar10/moco/checkpoint_last.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_folder', metavar='DIR', default='./results/cifar10/moco',
                    help='path to results')
parser.add_argument('--logs_folder', metavar='DIR', default='./results/cifar10/moco/logs',
                    help='path to tensorboard logs')

# running logistics
parser.add_argument('--save-freq', default=1, type=int, metavar='N',
                    help='epoch frequency of saving model')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

# optimizer
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')


def main():
    args = parser.parse_args()
    print(args)
    cfg = Config.fromfile(args.config_file)

    # checking GPU and showing infos
    if not torch.cuda.is_available():
        raise NotImplementedError("You need GPU!")
    print("Use GPU: {} for training".format(torch.cuda.current_device()))


    # create SPICE model
    # first get encoder from self-supervised model and load wieghts
    moco_model = moco.builder.MoCo(
        base_encoder=resnet18_cifar,
        dim=cfg.moco.moco_dim, K=cfg.moco.moco_k, m=cfg.moco.moco_m, T=cfg.moco.moco_t, mlp=cfg.moco.mlp, single_gpu=True)
    
    
    if not args.resume:
        # if it's the first time, load weights of the self-supervised model
        # when is resumed it is not necessary since they will be resumed for all spice model later
        loc = 'cuda:{}'.format(torch.cuda.current_device())
        checkpoint = torch.load(args.pretrained_self_supervised_folder, map_location=loc)
        moco_model.load_state_dict(checkpoint['state_dict'])

    # remove from encoder avgpool and fc layer in order to extract features
    moco_encoder = torch.nn.Sequential(*(list(moco_model.encoder_q.children())[:-2]))
    print(cfg.clustering_head[0][0])
    clustering_head = SemHeadMulti(multi_heads=cfg.clustering_head[0])

    model = SPICEModel(feature_model=moco_encoder, head=clustering_head)
    # logger.info(model)

    model.cuda()

    # optimizer only for Clustering Head
    optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr)


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(torch.cuda.current_device())
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Load similarity model

    cudnn.benchmark = True

    CIFAR10_normalization = transforms.Normalize(mean=cfg.dataset.normalization.mean, std=cfg.dataset.normalization.std)
    
    # weak augmentation
    weak_augmentation = transforms.Compose([
            transforms.RandomCrop(cfg.dataset.img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            CIFAR10_normalization
        ])

    # strong augmentation
    strong_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(cfg.dataset.img_size),
            Augment(4),
            transforms.ToTensor(),
            CIFAR10_normalization,
            Cutout(
                n_holes=1,
                length=16,
                random=True)])
    
    # training dataset will retrieve 2 images: one weakly augmented, one strong augmented
    train_dataset = CIFAR10Pair(root=args.dataset_folder, 
        train=True, 
        transform=dict(augmentation_1=weak_augmentation, augmentation_2=strong_augmentation),
        download=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.target_sub_batch_size, shuffle=False, # don't shuffle, the order is important because features are calculated using the original dataset
        num_workers=1, pin_memory=True, drop_last=True)
    
    # training dataset with non augmented images
    train_original_images = CIFAR10(root=args.dataset_folder, train=True, 
        transform=transforms.Compose([transforms.ToTensor(),
                                    CIFAR10_normalization]),
        download=True)

    train_original_images_loader = torch.utils.data.DataLoader(
        train_original_images, batch_size=cfg.target_sub_batch_size, shuffle=False, # don't shuffle, the order is important because features are calculated using the original dataset
        num_workers=1, pin_memory=True, drop_last=True)

    test_dataset = CIFAR10(root=args.dataset_folder, train=False, 
        transform=transforms.Compose([transforms.ToTensor(),
                                    CIFAR10_normalization]), 
        download=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.test_batch_size, shuffle=False, num_workers=1)

    best_acc = -2
    best_nmi = -1
    best_ari = -1
    best_head = -1
    best_epoch = -1
    min_loss = 1e10
    loss_head = -1
    loss_acc = -2
    loss_nmi = -1
    loss_ari = -1
    loss_epoch = -1

    train_writer = SummaryWriter(args.logs_folder + "/train")
    test_writer = SummaryWriter(args.logs_folder + "/test")

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, train_original_images_loader, model, optimizer, epoch, cfg, args)

        if (epoch+1) % args.save_freq == 0:
            # remove old checkpoint. I not overwrite because if something goes wrong the one before
            # will be in the bin and could be restored
            if os.path.exists('{}/checkpoint_last.pth.tar'.format(args.save_folder)):
                os.rename('{}/checkpoint_last.pth.tar'.format(args.save_folder), '{}/checkpoint_last-1.pth.tar'.format(args.save_folder))

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_last.pth.tar'.format(args.save_folder))

            if (epoch+1) == args.epochs:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename='{}/checkpoint_final.pth.tar'.format(args.save_folder))
            
            # start evaluation

            model.eval()

            loss_fn = nn.CrossEntropyLoss()
            num_heads = len(cfg.clustering_head)
            gt_labels = []
            pred_labels = []
            scores_all = []
            accs = []
            aris = []
            nmis = []
            features_all = []
            for h in range(num_heads):
                pred_labels.append([])
                scores_all.append([])

            # extract features and cluster predictions from test dataset
            for _, (images, labels) in enumerate(test_loader):
                images = images.cuda(non_blocking=True)

                with torch.no_grad():
                    features = model.extract_only_features(images)
                    scores = model.sem(images)

                features_all.append(features)

                assert len(scores) == num_heads
                for h in range(num_heads):
                    # for every image take indices of the best cluster predicted
                    pred_idx = scores[h].argmax(dim=1)
                    pred_labels[h].append(pred_idx)
                    scores_all[h].append(scores[h])

                gt_labels.append(labels)

            gt_labels = torch.cat(gt_labels).long().cpu().numpy()
            features_all = torch.cat(features_all, dim=0)
            features_all = features_all.cuda(non_blocking=True)
            losses = []

            for h in range(num_heads):
                scores_all[h] = torch.cat(scores_all[h], dim=0)
                pred_labels[h] = torch.cat(pred_labels[h], dim=0)

            with torch.no_grad():
                prototype_samples_indices, gt_cluster_labels = model.sim2sem(features_all, scores_all, epoch)

            for h in range(num_heads):
                pred_labels_h = pred_labels[h].long().cpu().numpy()

                pred_scores_select = scores_all[h][prototype_samples_indices[h].cpu()]
                gt_labels_select = gt_cluster_labels[h]
                loss = loss_fn(pred_scores_select.cpu(), gt_labels_select)

                try:
                    acc = calculate_acc(pred_labels_h, gt_labels)
                except:
                    acc = -1

                nmi = calculate_nmi(pred_labels_h, gt_labels)

                ari = calculate_ari(pred_labels_h, gt_labels)

                accs.append(acc)
                nmis.append(nmi)
                aris.append(ari)

                losses.append(loss.item())

            accs = np.array(accs)
            nmis = np.array(nmis)
            aris = np.array(aris)
            losses = np.array(losses)

            best_acc_real = accs.max()
            head_real = np.where(accs == best_acc_real) # return array of indices of elements that satisfy the condition
            head_real = head_real[0][0]     # select the index value
            best_nmi_real = nmis[head_real]
            best_ari_real = aris[head_real]
            print("Real: ACC: {}, NMI: {}, ARI: {}, head: {}".format(best_acc_real, best_nmi_real, best_ari_real, head_real))
            test_writer.add_scalar('Cluster Accuracy/real',
                best_acc_real,
                epoch)
            test_writer.add_scalar('Cluster NMI/real',
                best_nmi_real,
                epoch)
            test_writer.add_scalar('Cluster ARI/real',
                best_ari_real,
                epoch)

            head_loss = np.where(losses == losses.min())[0]
            head_loss = head_loss[0]
            best_acc_loss = accs[head_loss]
            best_nmi_loss = nmis[head_loss]
            best_ari_loss = aris[head_loss]
            print("Loss: ACC: {}, NMI: {}, ARI: {}, head: {}".format(best_acc_loss, best_nmi_loss, best_ari_loss, head_loss))
            test_writer.add_scalar('Cluster Accuracy/loss',
                best_acc_loss,
                epoch)
            test_writer.add_scalar('Cluster NMI/loss',
                best_nmi_loss,
                epoch)
            test_writer.add_scalar('Cluster ARI/loss',
                best_ari_loss,
                epoch)

            if best_acc_real > best_acc:
                best_acc = best_acc_real
                best_nmi = best_nmi_real
                best_ari = best_ari_real
                best_epoch = epoch
                best_head = np.array(accs).argmax()

                state_dict = model.state_dict()
                state_dict_save = {}
                for k in list(state_dict.keys()):
                    if not k.startswith('module.head'):
                        state_dict_save[k] = state_dict[k]
                    # print(k)
                    if k.startswith('module.head.head_{}'.format(best_head)):
                        state_dict_save['module.head.head_0.{}'.format(k[len('module.head.head_{}.'.format(best_head))::])] = state_dict[k]

                torch.save(state_dict_save, '{}/checkpoint_best.pth.tar'.format(args.save_folder))
                # save_checkpoint({
                #     'epoch': epoch + 1,
                #     'state_dict': model.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                # }, is_best=False, filename='{}/checkpoint_best.pth.tar'.format(cfg.results.output_dir))

            if min_loss > losses.min():
                min_loss = losses.min()
                loss_head = head_loss
                loss_epoch = epoch
                loss_acc = best_acc_loss
                loss_nmi = best_nmi_loss
                loss_ari = best_ari_loss

                state_dict = model.state_dict()
                state_dict_save = {}
                for k in list(state_dict.keys()):
                    if not k.startswith('module.head'):
                        state_dict_save[k] = state_dict[k]
                    # print(k)
                    if k.startswith('module.head.head_{}'.format(loss_head)):
                        state_dict_save['module.head.head_0.{}'.format(k[len('module.head.head_{}.'.format(loss_head))::])] = state_dict[k]

                torch.save(state_dict_save, '{}/checkpoint_select.pth.tar'.format(args.save_folder))

            model.train()

            print("FINAL -- Best ACC: {}, Best NMI: {}, Best ARI: {}, epoch: {}, head: {}".format(best_acc, best_nmi, best_ari, best_epoch, best_head))
            print("FINAL -- Select ACC: {}, Select NMI: {}, Select ARI: {}, epoch: {}, head: {}".format(loss_acc, loss_nmi, loss_ari, loss_epoch, loss_head))


def train(train_loader, train_original_images_loader, model, optimizer, epoch, cfg, args):
    info = []
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    info.append(batch_time)
    info.append(data_time)
    num_heads = cfg.num_head
    # clustering losses
    losses = []
    for h in range(num_heads):
        losses_h = AverageMeter('Loss_{}'.format(h), ':.4e')
        losses.append(losses_h)
        info.append(losses_h)
    lr = AverageMeter('lr', ':.6f')
    lr.update(optimizer.param_groups[0]["lr"])
    info.append(lr)

    progress = ProgressMeter(
        len(train_loader),
        info,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    target_sub_batch_size = cfg.target_sub_batch_size
    batch_size = cfg.batch_size
    train_sub_batch_size = cfg.train_sub_batch_size

    num_repeat = cfg.num_repeat

    # number of training images
    num_imgs_all = len(train_loader.dataset)

    # // is floor division
    iters_end = batch_size // target_sub_batch_size
    num_iters_l = num_imgs_all // batch_size
    
    # for every iteration on minibatches in dataset
    for ii in range(num_iters_l):
        end = time.time()

        # E-Step based on SPICE paper
        # model in eval mode
        model.eval()

        # clustering scores, for each clustering head
        scores = []
        for h in range(num_heads):
            scores.append([])

        strong_augmented_images_all = []
        original_images_features_all = []

        # extract features of original images, 
        print("Extracting features from original images")
        for o, (original_images, _) in enumerate(train_original_images_loader):
            # First branch: extract features from original images
            # if o == 8:
            #     fig = plt.figure()
            #     plt.imshow(original_images[o].numpy().transpose([1, 2, 0]))
            #     plt.savefig("original_img.png")

            original_images = original_images.cuda(non_blocking=True)
            with torch.no_grad():
                original_images_features = model.extract_only_features(original_images)
            
            original_images_features_all.append(original_images_features)
            
            if len(original_images_features_all) >= iters_end:
                break

        # calculate clustering scores from weakly augmented images
        print("Calculating clustering scores from weakly augmented images")
        for oo, (weak_augmented_images, strong_augmented_images) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            
            # if oo == 8:
            #     fig = plt.figure()
            #     plt.imshow(weak_augmented_images[oo].numpy().transpose([1, 2, 0]))
            #     plt.savefig("weakly_augmented.png")

            # Second branch: get scores for each sample belonging to clusters
            # Select samples and estimate the ground-truth relationship between samples.
            weak_augmented_images = weak_augmented_images.cuda(non_blocking=True)
            with torch.no_grad():
                # returns the probabilities of the features from clustering heads using softmax
                # list of len == clustering head, every item of list is a NxK tensor
                scores_nl = model.sem(weak_augmented_images)

            assert num_heads == len(scores_nl)

            for h in range(num_heads):
                # accumulate for every clustering head the scores for all the dataset
                scores[h].append(scores_nl[h].detach())

            strong_augmented_images_all.append(strong_augmented_images)

            if len(strong_augmented_images_all) >= iters_end:
                break

        # transform list(list(scores) into list(tensor(scores(DxK)))
        for h in range(num_heads):
            scores[h] = torch.cat(scores[h], dim=0)

        # transform list into tensor
        strong_augmented_images_all = torch.cat(strong_augmented_images_all)
        # transform list into tensor
        original_images_features_all = torch.cat(original_images_features_all)

        original_images_features_all = original_images_features_all.cuda(non_blocking=True)#.to(torch.float32)
        
        print("Calculating prototypes and cluster ground truth")
        # get prototype samples and clusters ground truth: 
        # for each cluster are proposed some images to be classified in, 
        # it return also the indices of the relative cluster of the prototype since the information of 
        # how many images are in a cluster is computed inside the model
        # tensor [K*num images per cluster], tensor [K*num images per cluster]
        prototype_samples_indices, gt_cluster_labels = model.sim2sem(original_images_features_all, scores, epoch)


        # M-Step based on SPICE paper
        strong_augmented_images_prototypes = []
        for h in range(num_heads):
            # take the strong augmented version of the prototypes selected in the previous step
            strong_augmented_images_prototypes.append(strong_augmented_images_all[prototype_samples_indices[h], :, :, :])
        
        num_images = strong_augmented_images_prototypes[0].shape[0]

        # Train the clustering heads with the generated ground truth
        model.train()

        # create list [0, 1, 2, ... num_images-1]
        images_prototypes_indices = list(range(num_images))
        # Select a set of images for training.

        num_train = num_images

        train_sub_iters = num_train // train_sub_batch_size

        print("Training clustering heads with previously extracted prototypes")
        for n in range(num_repeat):
            random.shuffle(images_prototypes_indices)

            for i in range(train_sub_iters):
                # variables to decide which portion of images to take
                start_idx = i * train_sub_batch_size
                end_idx = min((i + 1) * train_sub_batch_size, num_train)
                images_prototypes_indices_i = images_prototypes_indices[start_idx:end_idx]

                imgs_i = []
                targets_i = []

                for h in range(num_heads):
                    # take a portion of images and relative target (in which cluster should be classified)
                    imgs_i.append(strong_augmented_images_prototypes[h][images_prototypes_indices_i, :, :, :].cuda(non_blocking=True))
                    targets_i.append(gt_cluster_labels[h][images_prototypes_indices_i].cuda(non_blocking=True))

                clutering_heads_losses = model.loss(imgs_i, targets_i)

                loss = sum(loss for loss in clutering_heads_losses.values())
                loss_mean = loss / num_heads

                optimizer.zero_grad()
                loss_mean.backward()
                optimizer.step()

                for h in range(num_heads):
                    # measure accuracy and record loss
                    losses[h].update(clutering_heads_losses['head_{}'.format(h)].item(), imgs_i[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ii % args.save_freq == 0:
            progress.display(ii)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
