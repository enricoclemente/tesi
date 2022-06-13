#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from enum import unique
import math
import os
import shutil
import time
import sys
from xmlrpc.client import Boolean
sys.path.insert(0, './')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data


from SPICE.spice.config import Config
from torch.utils.tensorboard import SummaryWriter

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
parser.add_argument("--config_file", default="./experiments_config_example.py", metavar="FILE",
                    help="path to config file", type=str)
# arguments for saving and resuming                  
parser.add_argument('--dataset_folder', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('--resume', default='./results/checkpoints/checkpoint_last.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_folder', metavar='DIR', default='./results/checkpoints/',
                    help='path to save checkpoints')
parser.add_argument('--logs_folder', metavar='DIR', default='./results/logs',
                    help='path to save tensorboard logs')

# running logistics
parser.add_argument('--save-freq', default=1, type=int, metavar='N',
                    help='epoch frequency of saving model and plotting')
parser.add_argument('--validation-freq', default=5, type=int,
                    help='epoch frequency of validating the model' )
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')


# knn test hyperparameters
parser.add_argument('--knn_test', default=False, type=Boolean, help='enable kNN test')
parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')


def main():
    args = parser.parse_args()
    print("train moco started with params")
    print(args)
    cfg = Config.fromfile(args.config_file)

    # setting of save_folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    # setting of logs_folder
    if not os.path.exists(args.logs_folder):
        os.makedirs(args.logs_folder)

    # checking GPU and showing infos
    if not torch.cuda.is_available():
        raise NotImplementedError("You need GPU!")
    print("Use GPU: {} for training".format(torch.cuda.current_device()))

    torch.cuda.set_device(torch.cuda.current_device())

    
    base_encoder = models.resnet18
    pair_train_dataset = None
    validation_test_dataset = None
    validation_train_dataset = None
    dataset_normalization = transforms.Normalize(mean=cfg.dataset.normalization.mean, std=cfg.dataset.normalization.std)
    
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    # https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/data_aug/contrastive_learning_dataset.py#L8
    mocov2_augmentation = [
        transforms.RandomResizedCrop(cfg.dataset.img_size, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([SPICE.moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        dataset_normalization
        ]
    if cfg.dataset.dataset_name == 'cifar10':
        # resnet18_cifar which is an implementation adapted for CIFAR10
        base_encoder = resnet18_cifar

        # creating CIFAR10 train  dataset from custom CIFAR10Pair class 
        # which gave pair augmentation of image
        pair_train_dataset = CIFAR10Pair(root=args.dataset_folder, train=True, 
            transform=transforms.Compose(mocov2_augmentation), 
            download=True)

        # creating CIFAR10 datasets for knn test
        validation_test_dataset = CIFAR10(root=args.dataset_folder, train=False, 
            transform=transforms.Compose([transforms.ToTensor(),
                                        dataset_normalization]), 
            download=True)
        validation_train_dataset = CIFAR10(root=args.dataset_folder, train=True, 
            transform=transforms.Compose([transforms.ToTensor(),
                                        dataset_normalization]),  
            download=True)

    elif cfg.dataset.dataset_name == 'socialprofilepictures':
        # base resnet18 encoder since using images of the same size of ImageNet
        base_encoder = models.resnet18


        # creating SPP train dataset 
        # which gave pair augmentation of image
        pair_train_dataset = SocialProfilePicturesPair(version=cfg.dataset.version, randomize_metadata=cfg.dataset.randomize_metadata, root=args.dataset_folder, split="train", 
                                    transform=transforms.Compose(
                                                [PadToSquare() if cfg.dataset.train_padding else DoNothing(),    # apply padding to make images squared without
                                                transforms.Resize([cfg.dataset.img_size, cfg.dataset.img_size]) if cfg.dataset.train_resize else DoNothing()] +
                                                mocov2_augmentation))

        # creating SPP datasets for knn test
        validation_train_dataset = SocialProfilePictures(version=cfg.dataset.version, randomize_metadata=cfg.dataset.randomize_metadata, root=args.dataset_folder, split="train", 
                                    transform=transforms.Compose(
                                                [PadToSquare() if cfg.dataset.test_padding else DoNothing(),    
                                                transforms.Resize([cfg.dataset.img_size, cfg.dataset.img_size]) if cfg.dataset.test_resize else DoNothing(),
                                                transforms.RandomResizedCrop(cfg.dataset.img_size,),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                dataset_normalization]))
        validation_test_dataset = SocialProfilePictures(version=cfg.dataset.version, randomize_metadata=cfg.dataset.randomize_metadata, root=args.dataset_folder, split="val", 
                                    transform=transforms.Compose( 
                                                [PadToSquare() if cfg.dataset.test_padding else DoNothing(),    
                                                transforms.Resize([cfg.dataset.img_size, cfg.dataset.img_size]) if cfg.dataset.test_resize else DoNothing(),
                                                transforms.ToTensor(),
                                                dataset_normalization]))
        
    else:
        raise NotImplementedError("You must choose a valid dataset!")

    # creating dataset loaders for train
    pair_train_loader = torch.utils.data.DataLoader(
        pair_train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=1, 
        pin_memory=True, drop_last=True)
    # creating datasets loaders for validation
    validation_train_loader = torch.utils.data.DataLoader(
        validation_train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=1, 
        pin_memory=True)
    validation_test_loader = torch.utils.data.DataLoader(
        validation_test_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=1, 
        pin_memory=True)

    # creating model MoCo 
    model = SPICE.moco.builder.MoCo(
        base_encoder=base_encoder,
        dim=cfg.moco.moco_dim, K=cfg.moco.moco_k, m=cfg.moco.moco_m, T=cfg.moco.moco_t, mlp=cfg.moco.mlp, query_encoder_pretrained=cfg.moco.query_encoder_pretrained, key_encoder_pretrained=cfg.moco.key_encoder_pretrained)
    print(model)
    model = model.cuda()
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), cfg.training.lr,
                                momentum=cfg.optimizer.momentum,
                                weight_decay=cfg.optimizer.wd)

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

    # tensorboard plotter
    train_writer = SummaryWriter(args.logs_folder + "/training")
    if args.knn_test: 
        knn_writer = SummaryWriter(args.logs_folder + "/validation_knn_test")
    linear_classifier_train_writer = SummaryWriter(args.logs_folder + "/validation_linear_classifier(LDA)_train")
    linear_classifier_test_writer = SummaryWriter(args.logs_folder + "/validation_linear_classifier(LDA)_test")

    # checking encoder performances before training starts only the first time
    if args.start_epoch == 0:
        print("Linear Classifier first check")
        linear_classifier_lda_test(model.encoder_q, validation_train_loader, validation_test_loader, 0, linear_classifier_train_writer, linear_classifier_test_writer, args, cfg)

    best_acc = 0.0
    for epoch in range(args.start_epoch, cfg.training.epochs):

        # lr scheduling
        new_lr = adjust_learning_rate(optimizer, epoch, cfg)
        train_writer.add_scalar('Learning rate/lr decay', new_lr, epoch)
        
        # training
        print("Training epoch {}".format(epoch))
        metrics = train(pair_train_loader, model, criterion, optimizer, epoch, train_writer, args)
       
        if (epoch+1) % args.validation_freq == 0 and epoch != 0:
            # knn test
            if args.knn_test:
                print("KNN test for epoch {}".format(epoch))
                metrics['knn_test_acc@1'] = knn_test(model.encoder_q, validation_train_loader, validation_test_loader, epoch, knn_writer, args)

            # validate with linear classifier LDA
            print("Linear Classifier validation for epoch {}".format(epoch))
            metrics['linear_classifier_train_loss'], metrics['linear_classifier_train_acc@1'], metrics['linear_classifier_train_f1_score_micro'], metrics['linear_classifier_train_f1_score_macro'], metrics['linear_classifier_train_f1_score_weighted'], metrics['linear_classifier_train_f1_score_samples'], metrics['linear_classifier_test_loss'], metrics['linear_classifier_test_acc@1'], metrics['linear_classifier_test_f1_score_micro'], metrics['linear_classifier_test_f1_score_macro'], metrics['linear_classifier_test_f1_score_weighted'], metrics['linear_classifier_test_f1_score_samples'] = linear_classifier_lda_test(model.encoder_q, validation_train_loader, validation_test_loader, epoch, linear_classifier_train_writer, linear_classifier_test_writer, args, cfg)

        if best_acc < metrics['training_acc@1']:
            best_acc = metrics['training_acc@1']
            # save the epoch with the best accuracy
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet18',
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'metrics' : metrics,
            }, is_best=False, filename='{}/checkpoint_best.pth.tar'.format(args.save_folder))

        if (epoch+1) % args.save_freq == 0:
            # remove old checkpoint. I not overwrite because if something goes wrong the one before
            # will be in the bin and could be restored
            if os.path.exists('{}/checkpoint_last.pth.tar'.format(args.save_folder)):
                os.rename('{}/checkpoint_last.pth.tar'.format(args.save_folder), '{}/checkpoint_last-1.pth.tar'.format(args.save_folder))

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet18',
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'metrics' : metrics,
            }, is_best=False, filename='{}/checkpoint_last.pth.tar'.format(args.save_folder))


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

        if epoch == 0 and i < 3:
            # checking first images
            plt.imshow(img1[0].numpy().transpose([1, 2, 0]) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])
            plt.savefig("{}/training_img{}_aug1.png".format(args.save_folder, i))
            plt.close()
            plt.imshow(img2[0].numpy().transpose([1, 2, 0]) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])
            plt.savefig("{}/training_img{}_aug2.png".format(args.save_folder, i)) 
            plt.close()      
        
        img1 = img1.cuda(non_blocking=True)
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
    writer.add_scalar('Training Loss/epoch loss',
                losses.get_avg(),
                epoch)
    writer.add_scalar('Training Accuracy/epoch top1 accuracy',
                top1.get_avg(),
                epoch)
    writer.add_scalar('Training Accuracy/epoch top5 accuracy',
                top5.get_avg(),
                epoch)
    writer.add_scalar('Training Time/epoch time',
                batch_time.get_sum(),
                epoch)        
    metrics = {
            "training_loss" : losses.get_avg(),
            "training_acc@1": top1.get_avg(),
            "training_acc@5": top5.get_avg()
        }
    return metrics


# test using a knn monitor
def knn_test(model, memory_data_loader, test_data_loader, epoch, writer, args):
    model.eval()
    classes = len(memory_data_loader.dataset.classes) #get number of classes
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    start = time.time()
    with torch.no_grad():
        # generate feature bank from train dataset
        print("Collecting features")
        for data, target in memory_data_loader:
            feature = model(data.cuda(non_blocking=True)) # for every sample in the batch let predict features NxC tensor
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)    # create list of features [tensor1 (NxC), tensor2 (NxC), tensorM (NxC)] where M is the number of minibatches

        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()  # first concatenates all features tensors [NxC],[NxC],... 
                                                                        # then transpose --> [CxN],[CxN], ... --> Cx(N*number of minibatches) contiguous is for the memory format
        
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)   # tensor of labels (N*number of minibatches)

        # loop test data to predict the label by weighted knn search 
        print("Calculating accuracy")
        for i, (data, target) in enumerate(test_data_loader):   # iterate on test dataset
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = model(data)   # NxC
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            if i % args.print_freq == 0:
                print('KNN Test Epoch: [{}][{}/{}] Acc@1:{:.2f}%'.format(epoch, i, len(test_data_loader), total_top1 / total_num * 100))
    
    end = time.time()
    # statistics to be written at the end of every epoch
    writer.add_scalar('KNN Test Accuracy/epoch top1 accuracy',
                total_top1 / total_num * 100,
                epoch)     
    writer.add_scalar('KNN Test Time/epoch time',
                end - start,
                epoch) 
    print('KNN Test Epoch: [{}] Acc@1:{:.2f}%'.format(epoch, total_top1 / total_num * 100))
    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> NxD(dim of train dataset)
    sim_matrix = torch.mm(feature, feature_bank)
    # get top knn_k values for every element of sim_matrix and the indices where in each tensor they are located
    # in other words, for every element in the batch get the top sim scores
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1) # Nxknn_k tensor

    # get the labels for the topk values found before 
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)    # Nxknn_k

    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device) # N*knn_k x classes
    
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0) # N*knn_k x classes

    # weighted score ---> Nxclasses
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


# function to make validation training a linear classifier
def linear_classifier_lda_test(model, train_loader, test_loader, epoch, train_writer, test_writer, args, cfg):
    # copying model parameters in new encoder
    encoder = models.resnet18(num_classes=cfg.moco.moco_dim)
    state_dict = dict()
    for key in model.state_dict():
        if not key.startswith("fc"):
            state_dict[key] = model.state_dict()[key]
    
    msg = encoder.load_state_dict(state_dict, strict=False)
    # check if only fc layer parameters are missing
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    encoder = torch.nn.Sequential(*(list(encoder.children())[:-1])).cuda()

    # freezing parameters and setting in eval mode
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    criterion = nn.CrossEntropyLoss()

    train_features = []
    train_targets = []
    test_features = []
    test_targets = []
    start = time.time()

    print("Extracting training features")
    train_features, train_targets = extract_features_targets(encoder, train_loader, normalize=False)

    print("Extracting test features")
    test_features, test_targets = extract_features_targets(encoder, test_loader, normalize=False)

    
    linear_classifier = LinearDiscriminantAnalysis()
    print("Fitting LDA")
    linear_classifier.fit(train_features, train_targets)

    print("Calculating metrics on train dataset")
    train_y_pred = linear_classifier.predict(train_features)
    train_y_probabilities = linear_classifier.predict_proba(train_features)

    train_loss = criterion(torch.tensor(train_y_probabilities), torch.tensor(train_targets))

    train_top1 = (train_y_pred == train_targets).sum().item() / len(train_targets) * 100
    
    train_f1_score_micro = f1_score(train_targets, train_y_pred, average='micro') * 100
    train_f1_score_macro = f1_score(train_targets, train_y_pred, average='macro') * 100
    train_f1_score_weighted = f1_score(train_targets, train_y_pred, average='weighted') * 100
    # train_f1_score_samples = f1_score(train_targets, train_y_pred, average='samples') * 100
    train_f1_score_samples = 0

    train_writer.add_scalar('Linear Classifier Loss/epoch loss',
            train_loss,
            epoch)    
    train_writer.add_scalar('Linear Classifier Accuracy/epoch top1 accuracy',
            train_top1,
            epoch)  
    train_writer.add_scalar('Linear Classifier F1 score/epoch f1 score micro',
            train_f1_score_micro,
            epoch) 
    train_writer.add_scalar('Linear Classifier F1 score/epoch f1 score macro',
            train_f1_score_macro,
            epoch)
    train_writer.add_scalar('Linear Classifier F1 score/epoch f1 score weighted',
            train_f1_score_weighted,
            epoch)
    train_writer.add_scalar('Linear Classifier F1 score/epoch f1 score samples',
            train_f1_score_samples,
            epoch)
    
    train_cf_matrix = confusion_matrix(train_targets, train_y_pred)
    train_df_cm = pd.DataFrame(train_cf_matrix/np.sum(train_cf_matrix) *10, 
                                index = [i for i in train_loader.dataset.classes],
                                columns = [i for i in train_loader.dataset.classes])
    plt.figure(figsize = (24,14))
    sn.heatmap(train_df_cm, annot=True)
    plt.savefig('{}/validation_train_confusion_matrix_epoch_{}.svg'.format(args.save_folder, epoch))
    plt.close()

    print('Linear Classifier Training Epoch: [{}] Loss:{:.2f} Acc@1:{:.2f}`%` F1score micro:{:.2f} F1score macro:{:.2f} F1score weighted:{:.2f} F1score samples:{:.2f}'.format(epoch, train_loss, train_top1, train_f1_score_micro, train_f1_score_macro, train_f1_score_weighted, train_f1_score_samples))
    
    print("Calculating metrics on test dataset")
    test_y_pred = linear_classifier.predict(test_features)
    test_y_probabilities = linear_classifier.predict_proba(test_features)

    test_loss = criterion(torch.tensor(test_y_probabilities), torch.tensor(test_targets))

    test_top1 = (test_y_pred == test_targets).sum().item() / len(test_targets) * 100
    
    test_f1_score_micro = f1_score(test_targets, test_y_pred, average='micro') * 100
    test_f1_score_macro = f1_score(test_targets, test_y_pred, average='macro') * 100
    test_f1_score_weighted = f1_score(test_targets, test_y_pred, average='weighted') * 100
    # test_f1_score_samples = f1_score(test_targets, test_y_pred, average='samples') * 100
    test_f1_score_samples = 0

    test_writer.add_scalar('Linear Classifier Loss/epoch loss',
            test_loss,
            epoch)    
    test_writer.add_scalar('Linear Classifier Accuracy/epoch top1 accuracy',
            test_top1,
            epoch)  
    test_writer.add_scalar('Linear Classifier F1 score/epoch f1 score micro',
            test_f1_score_micro,
            epoch)
    test_writer.add_scalar('Linear Classifier F1 score/epoch f1 score macro',
            test_f1_score_macro,
            epoch) 
    test_writer.add_scalar('Linear Classifier F1 score/epoch f1 score weighted',
            test_f1_score_weighted,
            epoch) 
    test_writer.add_scalar('Linear Classifier F1 score/epoch f1 score samples',
            test_f1_score_samples,
            epoch)

    test_cf_matrix = confusion_matrix(test_targets, test_y_pred)
    test_df_cm = pd.DataFrame(test_cf_matrix/np.sum(test_cf_matrix) *10, 
                                index = [i for i in test_loader.dataset.classes],
                                columns = [i for i in test_loader.dataset.classes])
    plt.figure(figsize = (24,14))
    sn.heatmap(test_df_cm, annot=True)
    plt.savefig('{}/validation_test_confusion_matrix_epoch_{}.svg'.format(args.save_folder, epoch))
    plt.close()

    print('Linear Classifier Test Epoch: [{}] Loss:{:.2f} Acc@1:{:.2f}`%` F1score micro:{:.2f} F1score macro:{:.2f} F1score weighted:{:.2f} F1score samples:{:.2f}'.format(epoch, test_loss, test_top1, test_f1_score_micro, test_f1_score_macro, test_f1_score_weighted, test_f1_score_samples))

    end = time.time()

    train_writer.add_scalar('Linear Classifier Time/epoch time',
            end - start,
            epoch) 

    return train_loss, train_top1, train_f1_score_micro, train_f1_score_macro, train_f1_score_weighted, train_f1_score_samples, test_loss, test_top1, test_f1_score_micro, test_f1_score_macro, test_f1_score_weighted, test_f1_score_samples


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


def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate based on schedule"""
    lr = cfg.training.lr
    if cfg.training.keep_lr:    # if you want to continue training after cosine schedule cycle completed 
                        # keeping fixed the last lr value
        lr = optimizer.param_groups[0]['lr']
    else:
        if cfg.training.cosine_lr_decay:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / cfg.training.epochs))
        else:  # stepwise lr schedule
            for milestone in cfg.training.schedule_lr_decay:
                lr *= 0.1 if epoch >= milestone else 1.

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


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
