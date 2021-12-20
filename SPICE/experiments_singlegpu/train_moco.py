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

# knn monitor
parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')



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
    # print(model)

    torch.cuda.set_device(torch.cuda.current_device())
    model = model.cuda()
    # print("New model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # print(model.state_dict()['queue'])
    # print(model.state_dict()['queue_ptr'])
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
            # print("Resume's state_dict:")
            # for param_tensor in model.state_dict():
            #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            #     print(param_tensor, "\t", model.state_dict()[param_tensor])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True  # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

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


    # creating CIFAR10 datasets for knn test, here we need only to apply simple normalization
    knn_test_dataset = CIFAR10(root=args.dataset_folder, train=False, 
        transform=transforms.Compose([transforms.ToTensor(),
                                    CIFAR10_normalization]), 
        download=True)
    knn_test_loader = torch.utils.data.DataLoader(
        knn_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, 
        pin_memory=True)

    memory_data = CIFAR10(root=args.dataset_folder, train=True, 
        transform=transforms.Compose([transforms.ToTensor(),
                                    CIFAR10_normalization]),  
        download=True)
    memory_loader = torch.utils.data.DataLoader(
        memory_data, batch_size=args.batch_size, shuffle=False, num_workers=1, 
        pin_memory=True)


    # tensorboard plotter
    train_writer = SummaryWriter(args.logs_folder + "/train")
    test_writer = SummaryWriter(args.logs_folder + "/test")
    knn_test_writer = SummaryWriter(args.logs_folder + "/knn_test")

    best_acc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        
        # train for one epoch
        metrics = train(train_loader, model, criterion, optimizer, epoch, train_writer, args)

        test_acc1, test_acc5 = test(test_loader, model, criterion, epoch, test_writer, args)
        metrics['test_acc@1'] = test_acc1
        metrics['test_acc@5'] = test_acc5

        knn_test_acc1 = knn_test(model.encoder_q, memory_loader, knn_test_loader, epoch, knn_test_writer, args)
        metrics['knn_test_acc@1'] = knn_test_acc1

        if best_acc < metrics['training_acc@1']:
            best_acc = metrics['training_acc@1']
            # save the epoch with the best accuracy
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet18',
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'metrics' : metrics,
            }, is_best=False, filename='{}/checkpoint_best_epoch_{}.pth.tar'.format(args.save_folder, epoch))

        if (epoch+1) % args.save_freq == 0:
            # remove old checkpoint. I not overwrite because if something goes wrong the one before
            # will be in the bin and could be restored
            if os.path.exists('{}/checkpoint_last.pth.tar'.format(args.save_folder)):
                os.rename('{}/checkpoint_last.pth.tar'.format(args.save_folder), '{}/checkpoint_last-1.pth.tar'.format(args.save_folder))
            # print("Save's state_dict:")
            # for param_tensor in model.state_dict():
            #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            #     print(param_tensor, "\t", model.state_dict()[param_tensor])

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
    for i, (img1, img2) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if epoch == 0 and i == 0:
            # first check
            print("batches dimensions")
            print(img1.size())
            print(img2.size())

        img1= img1.cuda(non_blocking=True)
        img2 = img2.cuda(non_blocking=True)

        # compute output
        output, target = model.forward_singlegpu(im_q=img1, im_k=img2)
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
            # print("writing minibatch on tensorboard")
            writer.add_scalar('Training Loss/minibatches loss',
                        losses.get_avg(),
                        epoch * len(train_loader) + i)
            writer.add_scalar('Training Accuracy/minibatches top1 accuracy',
                        top1.get_avg(),
                        epoch * len(train_loader) + i)
            writer.add_scalar('Training Accuracy/minibatches top5 accuracy',
                        top5.get_avg(),
                        epoch * len(train_loader) + i)

    progress.display(i)
    # statistics to be written at the end of every epoch
    # print("writing epoch metrics on tensorboard")
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
            "training_loss" : losses.get_avg(),
            "training_acc@1": top1.get_avg(),
            "training_acc@5": top5.get_avg()
        }
    return metrics


def test(test_data_loader, model, criterion, epoch, writer, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.eval()
    
    with torch.no_grad():
        for i, (img1, img2) in enumerate(test_data_loader):
            img1, img2 = img1.cuda(non_blocking=True), img2.cuda(non_blocking=True)

            output, target = model.forward_singlegpu_eval(im_q=img1, im_k=img2)
            
            # calculate accuracy and loss
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # update meters
            top1.update(acc1[0])
            top5.update(acc5[0])
            losses.update(loss)
            
            if i % args.print_freq == 0:
                print('Test Epoch: [{}][{}/{}] Loss:{} Acc@1:{:.2f}% Acc@5:{:.2f}%'.format(epoch, i, len(test_data_loader), losses.get_avg(), top1.get_avg(), top5.get_avg()))
    
    # statistics to be written at the end of every epoch
    writer.add_scalar('Training Loss/epoch loss',
                losses.get_avg(),
                epoch)
    writer.add_scalar('Training Accuracy/epoch top1 accuracy',
                top1.get_avg(),
                epoch)    
    writer.add_scalar('Training Accuracy/epoch top5 accuracy',
                top5.get_avg(),
                epoch)
    print('Test Epoch: [{}][{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'.format(epoch, i, len(test_data_loader), top1.get_avg(), top5.get_avg()))
    return top1.get_avg(), top5.get_avg()


# test using a knn monitor
def knn_test(model, memory_data_loader, test_data_loader, epoch, writer, args):
    model.eval()
    classes = len(memory_data_loader.dataset.classes) #get number of classes
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank from train dataset
        for data, target in memory_data_loader:
            feature = model(data.cuda(non_blocking=True)) # for every sample in the batch let predict features NxC tensor
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)    # create list of features [tensor1 (NxC), tensor2 (NxC), tensorM (NxC)] where M is the number of minibatches

        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()  # first concatenates all features tensors [NxC],[NxC],... 
                                                                        # then transpose --> [CxN],[CxN], ... --> Cx(N*number of minibatches) contiguous is for the memory format
        
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)   # tensor of labels (N*number of minibatches)

        # loop test data to predict the label by weighted knn search 
        for i, (data, target) in enumerate(test_data_loader):   # iterate on test dataset
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = model(data)   # NxC
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            if i % args.print_freq == 0:
                print('KNN Test Epoch: [{}][{}/{}] Acc@1:{:.2f}%'.format(epoch, i, len(test_data_loader), total_top1 / total_num * 100))
    
    # statistics to be written at the end of every epoch
    writer.add_scalar('Training Accuracy/epoch top1 accuracy',
                total_top1 / total_num * 100,
                epoch)     
    print('KNN Test Epoch: [{}][{}/{}] Acc@1:{:.2f}%'.format(epoch, i, len(test_data_loader), total_top1 / total_num * 100))
    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> Nx(dim of train dataset)
    sim_matrix = torch.mm(feature, feature_bank)
    # get top knn_k values for every element of sim_matrix and the indices where in each tensor they are located
    # in other words, for every element in the batch get the top sim scores
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1) # Nxknn_k tensor

    # get the labels for the topk values found before
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)    # Nxknn_k

    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device) # N*knn_k x classes
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


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
    print("new learning rate: {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)
        # Returns the k largest elements of the given input tensor along a given dimension.
        # in this case output is Nx(1+K) and topk return a tensor long 1+K
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(k*correct.shape[1]).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()