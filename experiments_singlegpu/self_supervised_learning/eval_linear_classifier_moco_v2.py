#!/usr/bin/env python
import argparse
import sys
import os
import math

sys.path.insert(0, './')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from experiments_singlegpu.datasets.SocialProfilePictures import SocialProfilePictures, SocialProfilePicturesPro
import torchvision.transforms as transforms
from experiments_singlegpu.datasets.utils.custom_transforms import PadToSquare
from experiments_singlegpu.datasets.utils.custom_transforms import DoNothing

import torchvision.models as models

from SPICE.spice.config import Config
from SPICE.spice.model.feature_modules.resnet_cifar import resnet18_cifar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from torch.utils.tensorboard import SummaryWriter
from experiments_singlegpu.self_supervised_learning.utils import extract_features_targets, extract_features_targets_indices
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support



parser = argparse.ArgumentParser(description='Evaluation for MoCo with Linear Classifier')
parser.add_argument("--config_file", default="./experiments_config_example.py", metavar="FILE",
                    help="path to config file (same used with moco training)", type=str) 
parser.add_argument('--dataset_folder', metavar='DIR', default='./datasets/cifar10',
                    help='path to dataset')
parser.add_argument('--model_path', type=str, default=None,
                    help='path to model (checkpoint) trained with moco')
parser.add_argument('--save_folder', metavar='DIR', default='./results/cifar10/moco',
                    help='path where to save results')

parser.add_argument('--batch-size', default=512, type=int,
                    help='Number of images in each mini-batch')



def main():  
    args = parser.parse_args()
    print("linear classifier started with params:")
    print(args)
    cfg = Config.fromfile(args.config_file)

    # setting of save_folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # checking GPU and showing infos
    if not torch.cuda.is_available():
        raise NotImplementedError("You need GPU!")
    print("Use GPU: {} for training".format(torch.cuda.current_device()))

    torch.cuda.set_device(torch.cuda.current_device())

    encoder = models.resnet18()
    train_dataset = None
    test_dataset = None
    dataset_normalization = transforms.Normalize(mean=cfg.dataset.normalization.mean, std=cfg.dataset.normalization.std)
    
    if cfg.dataset.dataset_name == 'cifar10':
        # resnet18_cifar which is an implementation adapted for CIFAR10
        encoder = resnet18_cifar()

        # CIFAR10 train  dataset
        train_dataset = CIFAR10(root=args.dataset_folder, train=True, 
                        transform=transforms.Compose([transforms.ToTensor(), dataset_normalization]), download=True)
        test_dataset = CIFAR10(root=args.dataset_folder, train=False, 
                        transform=transforms.Compose([transforms.ToTensor(), dataset_normalization]), download=True)

    elif cfg.dataset.dataset_name == 'socialprofilepictures':
        # base resnet18 encoder since using images of the same size of ImageNet
        encoder = models.resnet18(pretrained=True if not args.model_path else False)

        # SPP train dataset 
        train_dataset = SocialProfilePicturesPro(version=cfg.dataset.version, root=args.dataset_folder, split="train", randomize_metadata=cfg.dataset.randomize_metadata,
                                    transform=transforms.Compose([
                                                transforms.Resize([cfg.dataset.img_size, cfg.dataset.img_size]),
                                                transforms.RandomResizedCrop(cfg.dataset.img_size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                dataset_normalization]))

        # SPP test dataset
        test_dataset = SocialProfilePicturesPro(version=cfg.dataset.version, root=args.dataset_folder, split="test", randomize_metadata=cfg.dataset.randomize_metadata,
                                    transform=transforms.Compose([
                                                transforms.Resize([cfg.dataset.img_size, cfg.dataset.img_size]), 
                                                transforms.ToTensor(),
                                                dataset_normalization]))
    else:
        raise NotImplementedError("You must choose a valid dataset!")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    if args.model_path:
        if os.path.isfile(args.model_path):
            print("Loading previously trained model on MoCo")
            loc = 'cuda:{}'.format(torch.cuda.current_device())
            checkpoint = torch.load(args.model_path, map_location=loc)
            state_dict = dict()
            for key in checkpoint['state_dict']:
                if key.startswith("encoder_q"):
                    state_dict[key[10:]] = checkpoint['state_dict'][key]
            
            encoder.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError("=> no checkpoint found at '{}'".format(args.model_path))
    else:
        print("Model path not specified, if the dataset is SPP, pretrained model on ImageNet will be used")
    
    # remove the fc layer from encoder
    encoder = torch.nn.Sequential(*(list(encoder.children())[:-1])).cuda()
    # print(encoder)
    # freeze encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False
    
    train_features = []
    train_targets = []
    train_indices = []
    test_features = []
    test_targets = []
    test_indices = []

    print("Extracting training features")
    train_features, train_targets, train_indices = extract_features_targets_indices(encoder, train_loader, normalize=False)

    print("Extracting test features")
    test_features, test_targets, test_indices = extract_features_targets_indices(encoder, test_loader, normalize=False)


    train_confusion_matrix_save_folder = os.path.join(args.save_folder, "train_confusion_matrix")
    if not os.path.exists(train_confusion_matrix_save_folder):
        os.makedirs(train_confusion_matrix_save_folder)

    train_f1_score_save_folder = os.path.join(args.save_folder, "train_f1_score")
    if not os.path.exists(train_f1_score_save_folder):
        os.makedirs(train_f1_score_save_folder)
    
    train_false_negatives_save_folder = os.path.join(args.save_folder, "train_false_negatives")
    if not os.path.exists(train_false_negatives_save_folder):
        os.makedirs(train_false_negatives_save_folder)
    
    train_false_positives_save_folder = os.path.join(args.save_folder, "train_false_positives")
    if not os.path.exists(train_false_positives_save_folder):
        os.makedirs(train_false_positives_save_folder)

    test_confusion_matrix_save_folder = os.path.join(args.save_folder, "test_confusion_matrix")
    if not os.path.exists(test_confusion_matrix_save_folder):
        os.makedirs(test_confusion_matrix_save_folder)

    test_f1_score_save_folder = os.path.join(args.save_folder, "test_f1_score")
    if not os.path.exists(test_f1_score_save_folder):
        os.makedirs(test_f1_score_save_folder)
    
    test_false_positives_save_folder = os.path.join(args.save_folder, "test_false_positives")
    if not os.path.exists(test_false_positives_save_folder):
        os.makedirs(test_false_positives_save_folder)
    
    test_false_negatives_save_folder = os.path.join(args.save_folder, "test_false_negatives")
    if not os.path.exists(test_false_negatives_save_folder):
        os.makedirs(test_false_negatives_save_folder)

    linear_classifier = LinearDiscriminantAnalysis()

    print("Fitting LDA")
    linear_classifier.fit(train_features, train_targets)

    print("Calculating metrics on train dataset")
    train_test_lda("train", linear_classifier, train_dataset, train_features, train_targets, train_indices, train_confusion_matrix_save_folder, train_f1_score_save_folder, train_false_positives_save_folder, train_false_negatives_save_folder, args)
    
    print("Calculating metrics on test dataset")
    train_test_lda("test", linear_classifier, test_dataset, test_features, test_targets, test_indices, test_confusion_matrix_save_folder, test_f1_score_save_folder, test_false_positives_save_folder, test_false_negatives_save_folder, args)


def train_test_lda(split, model, dataset, features, targets, indices, confusion_matrix_save_folder, f1_score_save_folder, false_positives_save_folder, false_negatives_save_folder, args):

    predictions = model.predict(features)

    top1 = (predictions == targets).sum().item() / len(targets)
    
    precision_micro, recall_micro, f1_score_micro, support = precision_recall_fscore_support(targets, predictions, average='micro')
    precision_macro, recall_macro, f1_score_macro, support = precision_recall_fscore_support(targets, predictions, average='macro')
    precision_weighted, recall_weighted, f1_score_weighted, support = precision_recall_fscore_support(targets, predictions, average='weighted')
    precision_per_class, recall_per_class, f1_score_per_class, support = precision_recall_fscore_support(targets, predictions, average=None) 
    
    precision_per_class_weighted, recall_per_class_weighted, f1_score_per_class_weighted = [], [], []
    for c_i in dataset.classes_count:
        precision_per_class_weighted.append(precision_per_class[dataset.classes_map[c_i]] * (dataset.classes_count[c_i]/len(dataset)))
        recall_per_class_weighted.append(recall_per_class[dataset.classes_map[c_i]] * (dataset.classes_count[c_i]/len(dataset)))
        f1_score_per_class_weighted.append(f1_score_per_class[dataset.classes_map[c_i]] * (dataset.classes_count[c_i]/len(dataset)))

    plot_multiple_bar_chart(dataset.classes, "Precision recall f1 score per class", "scores", f1_score_save_folder, [precision_per_class, recall_per_class, f1_score_per_class], ["precision", "recall", "f1 score"])
    
    total_false_positives = [ 0 for y in range(len(dataset.classes)) ]
    total_false_positives_weighted = [ 0 for y in range(len(dataset.classes)) ]
    total_false_negatives = [ 0 for y in range(len(dataset.classes)) ]
    total_false_negatives_weighted = [ 0 for y in range(len(dataset.classes)) ]
    false_positives_per_class = [ [ 0 for y in range(len(dataset.classes)) ] for x in range(len(dataset.classes)) ]
    false_positives_per_class_weighted = [ [ 0 for y in range(len(dataset.classes)) ] for x in range(len(dataset.classes)) ]
    false_positives_images_per_class = [ [] for c in dataset.classes]
    false_negatives_per_class = [ [ 0 for y in range(len(dataset.classes)) ] for x in range(len(dataset.classes)) ]
    false_negatives_per_class_weighted = [ [ 0 for y in range(len(dataset.classes)) ] for x in range(len(dataset.classes)) ]   
    false_negatives_images_per_class = [ [] for c in dataset.classes]
    predictions_results = predictions == targets
    for i, cp in enumerate(predictions_results):
        if cp == False:
            false_positives_per_class[targets[i]][predictions[i]] += 1
            false_positives_images_per_class[targets[i]].append({"img_path": os.path.join(dataset.metadata[indices[i]]['img_folder'],dataset.metadata[indices[i]]['img_name']), "wrong_prediction": dataset.classes[predictions[i]]})
            total_false_positives[targets[i]] += 1
            false_negatives_per_class[predictions[i]][targets[i]] += 1
            false_negatives_images_per_class[predictions[i]].append({"img_path": os.path.join(dataset.metadata[indices[i]]['img_folder'],dataset.metadata[indices[i]]['img_name']), "correct_prediction": dataset.classes[targets[i]]})
            total_false_negatives[predictions[i]] += 1

    for i, c_i in enumerate(dataset.classes):
        total_false_positives_weighted[i] = round(total_false_positives[i] / dataset.classes_count[c_i] * 100, 2)
        total_false_negatives_weighted[i] = round(total_false_negatives[i] / dataset.classes_count[c_i] * 100, 2)
        plot_bar_chart(dataset.classes, false_positives_per_class[i], "{} false positives over classes".format(c_i), "score", false_positives_save_folder)
        plot_bar_chart(dataset.classes, false_negatives_per_class[i], "{} false negatives over classes".format(c_i), "score", false_negatives_save_folder)
        
        for j, c_j in enumerate(dataset.classes):
            false_positives_per_class_weighted[i][j] = round(false_positives_per_class[i][j] / dataset.classes_count[c_j] * 100, 2)
            false_negatives_per_class_weighted[i][j] = round(false_negatives_per_class[i][j] / dataset.classes_count[c_j] * 100, 2)
        plot_bar_chart(dataset.classes, false_positives_per_class_weighted[i], "{} false positives over classes weighted".format(c_i), "score", false_positives_save_folder)
        plot_bar_chart(dataset.classes, false_negatives_per_class_weighted[i], "{} false negatives over classes weighted".format(c_i), "score", false_negatives_save_folder)
        
    plot_bar_chart(dataset.classes, total_false_positives, "Total false positives per class", "score", false_positives_save_folder)
    plot_bar_chart(dataset.classes, total_false_positives_weighted, "Total false positives per class weighted", "score", false_positives_save_folder)

    plot_bar_chart(dataset.classes, total_false_negatives, "Total false negatives per class", "score", false_negatives_save_folder)
    plot_bar_chart(dataset.classes, total_false_negatives_weighted, "Total false negatives per class weighted", "score", false_negatives_save_folder)
    
    cf_matrix = confusion_matrix(targets, predictions)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, 
                                index = [i for i in dataset.classes],
                                columns = [i for i in dataset.classes])
    plt.figure(figsize = (24,14))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('{}/confusion_matrix.svg'.format(confusion_matrix_save_folder))
    plt.close()
    
    with open("{}/{}_results.txt".format(args.save_folder, split), 'a') as file:
        file.write('Linear Classifier {}'.format(split))
        file.write('\n\tAcc@1:{:.2f}'.format(top1))
        file.write('\n\tWith average=micro precision:{:.2f} recall:{:.2f} f1 score:{:.2f}'.format(precision_micro, recall_micro, f1_score_micro))
        file.write('\n\tWith average=macro precision:{:.2f} recall:{:.2f} f1 score:{:.2f}'.format(precision_macro, recall_macro, f1_score_macro))
        file.write('\n\tWith average=weighted precision:{:.2f} recall:{:.2f} f1 score:{:.2f}'.format(precision_weighted, recall_weighted, f1_score_weighted))        
        file.write('\n\tPrecision, recall, f1 score per class')
        for i, v in enumerate(dataset.classes):
            file.write('\n\t {}: {:.2f}\t{:.2f}\t{:.2f}'.format(v, precision_per_class[i], recall_per_class[i], f1_score_per_class[i]))
    
    for i, v in enumerate(dataset.classes):
        with open("{}/{}_false_positives.txt".format(false_positives_save_folder, v), 'a') as file:
            file.write("{} false positives for class: {}".format(split, v))
            for e in false_positives_images_per_class[i]:
                file.write('\n\t{}'.format(e, ))
        with open("{}/{}_false_negatives.txt".format(false_negatives_save_folder, v), 'a') as file:
            file.write("{} false negatives for class: {}".format(split, v))
            for e in false_negatives_images_per_class[i]:
                file.write('\n\t{}'.format(e, ))


def plot_bar_chart(x, y, title, ylabel, save_folder):
    plt.xticks(rotation=45, ha='right')
    rect = plt.bar(x, y)
    plt.bar_label(rect, padding=3)

    plt.gca().set(title=title, ylabel=ylabel)
    plt.tight_layout()

    bottom, top = plt.ylim()
    plt.ylim([bottom, top + top*0.2])
    
    plt.savefig("{}/{}.svg".format(save_folder, title.lower().replace(" ", "_")))
    plt.close()


def plot_multiple_bar_chart(x_labels, title, ylabel, save_folder, y_values, y_labels):
    
    x_values = np.array(range(0,len(x_labels)))

    width = 1.0 / len(y_values) - 0.1 * 1.0 / len(y_values)
    # create array with offsets for every bar
    widths = np.linspace(- width * (len(y_values)-1)/2, width * (len(y_values)-1)/2, len(y_values))

    plt.figure(title, figsize=(15, 6))

    plt.gca().set(title=title, ylabel=ylabel)
    for i,y in enumerate(y_values):
        rects = plt.bar(x_values + widths[i], y, width, label=y_labels[i])
        plt.bar_label(rects, fmt='%.2f', padding=3)
    
    plt.xticks(x_values, x_labels, rotation=45, ha='right')
    plt.legend()

    # bottom, top = plt.ylim()
    # plt.ylim([bottom, top + top*0.2])
    plt.tight_layout()
    plt.savefig("{}/{}.svg".format(save_folder, title.lower().replace(" ", "_")))
    plt.close()


if __name__ == '__main__':
    main()