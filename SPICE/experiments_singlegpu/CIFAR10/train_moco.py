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

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
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
    print("train moco started with params")
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
    print(model)

    torch.cuda.set_device(torch.cuda.current_device())
    model = model.cuda()
    

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

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
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True  # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

    # Data loading code
    CIFAR10_normalize_values = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    # https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/data_aug/contrastive_learning_dataset.py#L8
    augmentation = [
        transforms.RandomResizedCrop(32),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        CIFAR10_normalize_values
    ]
    
    # creating CIFAR10 train dataset
    train_dataset = CIFAR10(root=args.dataset_folder, train=True, 
        transform=moco.loader.TwoCropsTransform(transforms.Compose(augmentation)), 
        download=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, 
        pin_memory=True, drop_last=True)

    # check training dataset 
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # show images
    print("showing some training images")
    imshow(torchvision.utils.make_grid(images[0]))
    imshow(torchvision.utils.make_grid(images[1]))

    # creating CIFAR10 test dataset, here we need only to apply simple normalization
    test_dataset = CIFAR10(root=args.dataset_folder, train=False, 
        transform=transforms.Compose([transforms.ToTensor(),
                                    CIFAR10_normalize_values]), 
        download=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, 
        pin_memory=True, drop_last=False)

    # tensorboard plotter
    writer = SummaryWriter(args.logs_folder + "/" + args.run_id)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        
        # train for one epoch
        metrics = train(train_loader, model, criterion, optimizer, epoch, writer, args)


        if (epoch+1) % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet18',
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'metrics' : metrics,
            }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.save_folder, epoch))

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet18',
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'metrics' : metrics,
            }, is_best=False, filename='{}/checkpoint_last.pth.tar'.format(args.save_folder))

        if (epoch+1) == args.epochs:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet18',
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'metrics' : metrics,
            }, is_best=False, filename='{}/checkpoint_final.pth.tar'.format(args.save_folder))


def train(train_loader, model, criterion, optimizer, epoch, writer,args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if epoch == 1:
            # just want to display a pair of images in order to track training samples
            img1 = images[0][0] / 2 + 0.5     # unnormalize
            img2 = images[1][0] / 2 + 0.5     # unnormalize
            writer.add_image("Training examples/epoch 1 image1",img1)  
            writer.add_image("Training examples/epoch 1 image12",img2)  
        
        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item())
        top1.update(acc1[0])
        top5.update(acc5[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            print("writing minibatch on tensorboard")
            writer.add_scalar('Training Loss/minibatches loss',
                        losses.get_avg(),
                        epoch * len(train_loader) + i)
            writer.add_scalar('Training Accuracy/minibatches top1 accuracy',
                        top1.get_avg(),
                        epoch * len(train_loader) + i)
            writer.add_scalar('Training Accuracy/minibatches top5 accuracy',
                        top5.get_avg(),
                        epoch * len(train_loader) + i)

    # statistics to be written at the end of every epoch
    print("writing epoch metrics on tensorboard")
    writer.add_scalar('Training Loss/epoch loss',
                losses.get_avg(),
                epoch)
    writer.add_scalar('Training Accuracy/epoch top1 accuracy',
                top1.get_avg(),
                epoch)
    writer.add_scalar('Training Accuracy/epoch top5 accuracy',
                top5.get_avg(),
                epoch)
    writer.add_scalar('Training Time/batch time',
                batch_time.get_sum(),
                epoch)        
    metrics = {
            "loss" : losses.get_avg(),
            "top1": top1.get_avg(),
            "top5": top5.get_avg()
        }
    return metrics



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
    
    def get_avg(self):
        return self.avg
    
    def get_sum(self):
        return self.sum

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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(k*correct.shape[1]).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    main()
