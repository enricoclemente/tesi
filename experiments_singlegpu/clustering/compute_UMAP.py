#!/usr/bin/env python
import sys
import random
import os
import argparse
sys.path.insert(0, './')

import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision.models.resnet import resnet18
from experiments_singlegpu.datasets.SocialProfilePictures import SocialProfilePictures

import torchvision.transforms as transforms
from experiments_singlegpu.datasets.utils.custom_transforms import PadToSquare

import matplotlib.pyplot as plt
import cuml
import umap


parser = argparse.ArgumentParser(description='UMAP calculator')
parser.add_argument('--n_components', type=int, default=2,
                    help='number of components to execute manifold')
parser.add_argument('--features_folder', metavar='DIR', default='./results/cifar10/moco',
                    help='path to previously calculated features')
parser.add_argument('--save_folder', metavar='DIR', default='./results/cifar10/moco',
                    help='path to results')

def main():
    args = parser.parse_args()
    # set seed in order to reproduce always the same result
    # seed = 10
    # random.seed(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # setting of save_folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    dataset = SocialProfilePictures(root='/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets', split=['train','val','test'],
                    transform=transforms.Compose([PadToSquare(), transforms.Resize([225, 225]), transforms.ToTensor()]))

    if not (os.path.isfile("{}/features.npy".format(args.features_folder)) 
            and os.path.isfile("{}/targets.npy".format(args.features_folder))):
        print("No presaved features and targets found!")
        model = resnet18(pretrained=True)
        # removing fc and avg layer
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        for p in model.parameters():
            p.requires_grad = False

        print(model)
        model.cuda()


        loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=1, 
                    pin_memory=True, drop_last=False)

        print("Extracting features")
        features = []
        targets = []

        model.eval()

        with torch.no_grad():
            for i, (img, target) in enumerate(loader):
                img = img.cuda()

                feature = model.forward(img)
                feature = torch.flatten(feature, 1)
                feature = F.normalize(feature, dim=1)

                # collecting all features and targets
                features.append(feature.cpu())
                targets.append(target)
                
                # print(target[0].item())
                # exit()
                print("[{}]/[{}] batch iteration".format(i, len(loader)))
                # if i == 2:
                #     break

        features = torch.cat(features)
        features = features.numpy()

        targets = torch.cat(targets)
        targets = targets.numpy()

        np.save("{}/features.npy".format(args.save_folder), features)
        np.save("{}/targets.npy".format(args.save_folder), targets)
    else:
        print("Loading previously calculated features and targets")
        features = np.load("{}/features.npy".format(args.save_folder))
        targets = np.load("{}/targets.npy".format(args.save_folder))

    exit()

    # calculating UMAP
    if not os.path.isfile("{}/umap_{}_components.npy".format(args.save_folder, args.n_components)):
        print("Calculating UMAP")
        reducer = cuml.UMAP(
                            n_neighbors=30,
                            n_components=args.n_components,
                            min_dist=0.25,
                            n_epochs=1000
                        ).fit(features)
        
        # umap = reducer.fit_transform(features)
        # reducer = umap.UMAP(n_neighbors=20,
        #                     n_components=args.n_components,
        #                     min_dist=0.0).fit(features)
        embedding = reducer.transform(features)
        np.save("{}/umap_{}_components.npy".format(args.save_folder, args.n_components), embedding)
    else:
        print("Loading UMAP")
        embedding = np.load("{}/umap_{}_components.npy".format(args.save_folder, args.n_components))

    # plot only 2d results
    if args.n_components == 2:

        colors_per_class = {}
        for class_name in dataset.classes:
            colors_per_class[class_name] = list(np.random.choice(range(256), size=3))
        # extract x and y coordinates representing the positions of the images on T-SNE plot
        tx = embedding[:, 0]
        ty = embedding[:, 1]

        # initialize a matplotlib plot
        fig = plt.figure("UMAP", figsize=(20,15))

        ax = fig.add_subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        plots = list(range(len(dataset.classes)))
        # for every class, we'll add a scatter plot separately
        for class_index, class_name in enumerate(dataset.classes):
            # find the samples of the current class in the data
            indices = [i for i, l in enumerate(targets) if l == class_index]

            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            # convert the class color to matplotlib format
            color = np.array(colors_per_class[class_name], dtype=float) / 255

            # add a scatter plot with the corresponding color and label
            plots[class_index] = ax.scatter(current_tx, current_ty,  c=color, label=class_name)
        
        # build a legend using the labels we set previously
        ax.legend(bbox_to_anchor=(1.0, 1.0))
        legendFig = plt.figure("UMAP legend", figsize=(5,7))
        legendFig.legend(plots, dataset.classes, loc='center')

        # finally, show the plot
        fig.savefig('{}/UMAP.svg'.format(args.save_folder))
        legendFig.savefig('{}/UMAP_legend.svg'.format(args.save_folder))

        plt.close()

        # plotting 2d t-SNE at double dims
        fig = plt.figure("UMAP 2x", figsize=(40,30))

        ax = fig.add_subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        plots = list(range(len(dataset.classes)))
        # for every class, we'll add a scatter plot separately
        for class_index, class_name in enumerate(dataset.classes):
            # find the samples of the current class in the data
            indices = [i for i, l in enumerate(targets) if l == class_index]

            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            # convert the class color to matplotlib format
            color = np.array(colors_per_class[class_name], dtype=float) / 255

            # add a scatter plot with the corresponding color and label
            plots[class_index] = ax.scatter(current_tx, current_ty,  c=color, label=class_name)
        
        ax.legend(bbox_to_anchor=(1.0, 1.0))

        # finally, show the plot
        fig.savefig('{}/UMAP_2x.svg'.format(args.save_folder))

        plt.close()

        ############################################################
        # create figures but with black background
        plt.style.use('dark_background')

        # initialize a matplotlib plot
        fig = plt.figure("UMAP", figsize=(20,15))

        ax = fig.add_subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        plots = list(range(len(dataset.classes)))
        # for every class, we'll add a scatter plot separately
        for class_index, class_name in enumerate(dataset.classes):
            # find the samples of the current class in the data
            indices = [i for i, l in enumerate(targets) if l == class_index]

            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            # convert the class color to matplotlib format
            color = np.array(colors_per_class[class_name], dtype=float) / 255

            # add a scatter plot with the corresponding color and label
            plots[class_index] = ax.scatter(current_tx, current_ty,  c=color, label=class_name)
        
        # build a legend using the labels we set previously
        ax.legend(bbox_to_anchor=(1.0, 1.0))
        legendFig = plt.figure("UMAP legend", figsize=(5,7))
        legendFig.legend(plots, dataset.classes, loc='center')

        # finally, show the plot
        fig.savefig('{}/UMAP_black.svg'.format(args.save_folder))

        plt.close()

        # plotting 2d t-SNE at double dims
        fig = plt.figure("UMAP 2x", figsize=(40,30))

        ax = fig.add_subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        plots = list(range(len(dataset.classes)))
        # for every class, we'll add a scatter plot separately
        for class_index, class_name in enumerate(dataset.classes):
            # find the samples of the current class in the data
            indices = [i for i, l in enumerate(targets) if l == class_index]

            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            # convert the class color to matplotlib format
            color = np.array(colors_per_class[class_name], dtype=float) / 255

            # add a scatter plot with the corresponding color and label
            plots[class_index] = ax.scatter(current_tx, current_ty,  c=color, label=class_name)
        
        ax.legend(bbox_to_anchor=(1.0, 1.0))

        # finally, show the plot
        fig.savefig('{}/UMAP_2x_black.svg'.format(args.save_folder))

        plt.close()




    
# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

if __name__ == '__main__':
    main()
