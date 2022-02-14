#!/usr/bin/env python
import argparse
import math
import os
import random
import shutil
import time
import sys
import site
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
from code.SPICE.experiments_singlegpu.datasets.CIFAR10_custom import CIFAR10Pair

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
parser.add_argument('--pretrained_self_supervised_model', default='./results/cifar10/moco/checkpoint_last.pth.tar', type=str, metavar='PATH',
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
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')


def main():
    torch.set_printoptions(linewidth=150)

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
        dim=cfg.moco.moco_dim, K=cfg.moco.moco_k, m=cfg.moco.moco_m, T=cfg.moco.moco_t, mlp=cfg.moco.mlp)
    
    if not os.path.isfile(args.resume):
        print("Loading encoder's weights")
        # if it's the first time, load weights of the self-supervised model
        # when is resumed it is not necessary since they will be resumed for all spice model later
        loc = 'cuda:{}'.format(torch.cuda.current_device())
        checkpoint = torch.load(args.pretrained_self_supervised_model, map_location=loc)
        moco_model.load_state_dict(checkpoint['state_dict'])

    # print(moco_model)
    # remove from encoder avgpool and fc layer in order to extract features
    moco_encoder = torch.nn.Sequential(*(list(moco_model.encoder_q.children())[:-2]))
    # print(moco_encoder)

    # print(cfg.clustering_head[0][0])
    clustering_head = SemHeadMulti(multi_heads=cfg.clustering_head[0])

    model = SPICEModel(feature_model=moco_encoder, head=clustering_head)
    # print(model)
    
    model.cuda()

    # optimizer only for Clustering Head
    optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr, weight_decay=args.wd)


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
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

    # print(model.feature_module.state_dict()['5.1.bn2.weight'])

   
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
    train_original_images = CIFAR10(root=args.dataset_folder, 
        train=True, 
        transform=transforms.Compose([transforms.ToTensor(),
                                    CIFAR10_normalization]),
        download=True)

    train_original_images_loader = torch.utils.data.DataLoader(
        train_original_images, batch_size=cfg.target_sub_batch_size, shuffle=False, # don't shuffle, the order is important because features are calculated using the original dataset
        num_workers=1, pin_memory=True, drop_last=True)

    # test dataset
    test_dataset = CIFAR10(root=args.dataset_folder, train=False, 
        transform=transforms.Compose([transforms.ToTensor(),
                                    CIFAR10_normalization]), 
        download=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.test_batch_size, shuffle=False, 
        num_workers=1, pin_memory=True)

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
        train(train_loader, train_original_images_loader, model, optimizer, epoch, train_writer, cfg, args)

        if (epoch+1) % args.save_freq == 0:
            print("Saving checkpoint")
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
            print("Starting evaluation")
            model.eval()

            loss_fn = nn.CrossEntropyLoss()
            num_heads = cfg.num_head
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
                except AssertionError as msg:
                    print("Clustering Accuracy failed: ") 
                    print(msg)
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

            # plotting results for every clu head and avgs
            loss_avg = 0
            acc_avg = 0
            ari_avg = 0
            nmi_avg = 0
            for h in range(num_heads):
                test_writer.add_scalar('Cluster Loss/epoch loss head_{}'.format(h),
                    losses[h],
                    epoch)
                test_writer.add_scalar('Cluster ACC/epoch acc head_{}'.format(h),
                    accs[h],
                    epoch)
                test_writer.add_scalar('Cluster ARI/epoch ari head_{}'.format(h),
                    aris[h],
                    epoch)
                test_writer.add_scalar('Cluster NMI/epoch nmi head_{}'.format(h),
                    nmis[h],
                    epoch)    
                loss_avg += losses[h]
                acc_avg += accs[h]
                ari_avg += aris[h]
                nmi_avg += nmis[h]
            
            loss_avg= loss_avg / num_heads
            acc_avg = acc_avg / num_heads
            ari_avg = ari_avg / num_heads
            nmi_avg = nmi_avg / num_heads
            test_writer.add_scalar('Cluster Loss/epoch loss avg',
                loss_avg,
                epoch)
            test_writer.add_scalar('Cluster ACC/epoch acc avg',
                acc_avg,
                epoch)
            test_writer.add_scalar('Cluster ARI/epoch ari avg',
                ari_avg,
                epoch)
            test_writer.add_scalar('Cluster NMI/epoch nmi avg',
                nmi_avg,
                epoch)

            best_acc_real = accs.max()
            head_real = np.where(accs == best_acc_real) # return array of indices of elements that satisfy the condition
            head_real = head_real[0][0]     # select the index value
            best_nmi_real = nmis[head_real]
            best_ari_real = aris[head_real]
            print("Real: ACC: {}, NMI: {}, ARI: {}, head: {}".format(best_acc_real, best_nmi_real, best_ari_real, head_real))
                
            test_writer.add_scalar('Cluster ACC/best acc head',
                best_acc_real,
                epoch)
            test_writer.add_scalar('Cluster NMI/best acc head',
                best_nmi_real,
                epoch)
            test_writer.add_scalar('Cluster ARI/best acc head',
                best_ari_real,
                epoch)
            test_writer.add_scalar('Cluster Loss/best acc head',
                losses[head_real],
                epoch)
            test_writer.add_scalar('Cluster Head/best acc head',
                head_real,
                epoch)

            head_loss = np.where(losses == losses.min())[0]
            head_loss = head_loss[0]
            best_acc_loss = accs[head_loss]
            best_nmi_loss = nmis[head_loss]
            best_ari_loss = aris[head_loss]
            print("Loss: ACC: {}, NMI: {}, ARI: {}, head: {}".format(best_acc_loss, best_nmi_loss, best_ari_loss, head_loss))
            test_writer.add_scalar('Cluster ACC/best loss head',
                best_acc_loss,
                epoch)
            test_writer.add_scalar('Cluster NMI/best loss head',
                best_nmi_loss,
                epoch)
            test_writer.add_scalar('Cluster ARI/best loss head',
                best_ari_loss,
                epoch)
            test_writer.add_scalar('Cluster Loss/best loss head',
                losses[head_loss],
                epoch)
            test_writer.add_scalar('Cluster Head/best loss head',
                head_loss,
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


def train(train_loader, train_original_images_loader, model, optimizer, epoch, train_writer, cfg, args):
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

    # switch to train mode
    target_sub_batch_size = cfg.target_sub_batch_size
    batch_size = cfg.batch_size
    train_sub_batch_size = cfg.train_sub_batch_size

    num_repeat = cfg.num_repeat

    # number of training images
    num_imgs_all = len(train_loader.dataset)
    print("Total number of training images: {}".format(num_imgs_all))

    # // is floor division
    iters_per_batch = batch_size // target_sub_batch_size
    total_batches = num_imgs_all // batch_size

    progress = ProgressMeter(
        total_batches,
        info,
        prefix="Epoch: [{}]".format(epoch))
    
    # iterate all over the dataset batch_size images per time
    for b in range(total_batches):
        end = time.time()

        # E-Step based on SPICE paper
        # model in eval mode
        model.eval()

        # clustering scores, for each clustering head for the batch
        scores = []
        for h in range(num_heads):
            scores.append([])

        strong_augmented_images_all = []
        original_images_features_all = []


        # First branch: extract features from original images
        print("Extracting features from original images")
        show = cfg.show_images
        for i, (original_images, _) in enumerate(train_original_images_loader):
            # take images corresponding to the batch
            if i >= (b*iters_per_batch) and i < (b+1)*iters_per_batch:
                # show the first image of the batch
                if show:
                    plt.imshow(original_images[0].numpy().transpose([1, 2, 0]) * cfg.dataset.normalization.std + cfg.dataset.normalization.std)
                    plt.savefig("original_img_{}.png".format(i))
                    show = False

                original_images = original_images.cuda(non_blocking=True)
                with torch.no_grad():
                    # extract original images features using feature module
                    # tensor [N,F]
                    original_images_features = model.extract_only_features(original_images)
                
                original_images_features_all.append(original_images_features)
        

        # Second branch: get scores for each sample belonging to clusters using weakly augmented images
        print("Calculating clustering scores from weakly augmented images")
        show = cfg.show_images
        for i, (weak_augmented_images, strong_augmented_images) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # take images corresponding to the batch
            if i >= (b*iters_per_batch) and i < (b+1)*iters_per_batch:
                # show the first image of the batch
                if show:
                    plt.imshow(weak_augmented_images[0].numpy().transpose([1, 2, 0]) * cfg.dataset.normalization.std + cfg.dataset.normalization.std)
                    plt.savefig("weak_img_{}.png".format(i))
                    plt.imshow(strong_augmented_images[0].numpy().transpose([1, 2, 0]) * cfg.dataset.normalization.std + cfg.dataset.normalization.std)
                    plt.savefig("strong_img_{}.png".format(i))
                    show = False
                
                weak_augmented_images = weak_augmented_images.cuda(non_blocking=True)
                with torch.no_grad():
                    # returns the probabilities of the features from clustering heads using softmax
                    # list of len == clustering head, every item of list is a tensor [N,K]
                    scores_nl = model.sem(weak_augmented_images)

                assert num_heads == len(scores_nl)

                for h in range(num_heads):
                    # accumulate for every clustering head the scores for all the batch
                    scores[h].append(scores_nl[h].detach())

                # save strongly augmented images for later
                strong_augmented_images_all.append(strong_augmented_images)


        # transform list(list(scores)) into list(tensor(scores[B,K]))
        for h in range(num_heads):
            scores[h] = torch.cat(scores[h], dim=0)

        # transform list into tensor
        strong_augmented_images_all = torch.cat(strong_augmented_images_all)
        # print("Strong augmented images shape: {}".format(strong_augmented_images_all.size()))
        # transform list into tensor
        original_images_features_all = torch.cat(original_images_features_all)
        # print("Original images features shape: {}".format(original_images_features_all.size()))

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
        # print("Number of protoypes: {}".format(num_images))

        # Train the clustering heads with the generated ground truth
        model.train()

        # create list [0, 1, 2, ... num_images-1]
        images_prototypes_indices = list(range(num_images))
        
        # Select a set of images for training.
        num_train = num_images
        train_sub_iters = num_train // train_sub_batch_size

        print("Training clustering head with previously extracted prototypes")
        for n in range(num_repeat):
            cudnn.benchmark = True
            random.shuffle(images_prototypes_indices)
            # print("Repetition {}/{}. GPU usage".format(n, num_repeat))
            # print(torch.cuda.memory_summary(torch.cuda.current_device()))
            for i in range(train_sub_iters):
                
                # variables to decide which portion of images to take
                start_idx = i * train_sub_batch_size
                end_idx = min((i + 1) * train_sub_batch_size, num_train)
                images_prototypes_indices_i = images_prototypes_indices[start_idx:end_idx]

                imgs_i = []
                targets_i = []

                for h in range(num_heads):
                    # take a portion of images and relative target (in which cluster the images should be classified)
                    imgs_i.append(strong_augmented_images_prototypes[h][images_prototypes_indices_i, :, :, :].cuda(non_blocking=True))
                    targets_i.append(gt_cluster_labels[h][images_prototypes_indices_i].cuda(non_blocking=True))

                clustering_head_losses = model.loss(imgs_i, targets_i)

                loss = sum(loss for loss in clustering_head_losses.values())
                loss_mean = loss / num_heads

                optimizer.zero_grad()
                loss_mean.backward()
                optimizer.step()

                for h in range(num_heads):
                    # measure accuracy and record loss
                    losses[h].update(clustering_head_losses['head_{}'.format(h)].item(), imgs_i[0].size(0))

        cudnn.benchmark = False

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if b % args.save_freq == 0:
            progress.display(b)

            # avg loss of clustering heads
            loss_avg = 0
            for h in range(num_heads):
                head_loss = losses[h].get_avg()
                train_writer.add_scalar('Cluster Loss/epoch loss head_{}'.format(h),
                head_loss,
                epoch)
                loss_avg += head_loss

            loss_avg = loss_avg / num_heads
            train_writer.add_scalar('Cluster Loss/epoch loss avg',
                loss_avg,
                epoch)



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


if __name__ == '__main__':
    main()
