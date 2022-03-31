#!/usr/bin/env python
import sys
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
# import umap


parser = argparse.ArgumentParser(description='UMAP calculator')
parser.add_argument('--features_folder', metavar='DIR', default='./features',
                    help='path to previously calculated features')
parser.add_argument('--save_folder', metavar='DIR', default='./results',
                    help='path to results')
parser.add_argument('--n_components', type=int, default=2,
                    help='umap number of components')
parser.add_argument('--n_neighbors', type=int, default=15,
                    help='umap number of neighbors')
parser.add_argument('--min_dist', type=float, default=0.0,
                    help='umap minimum distance')
parser.add_argument('--n_epochs', type=int, default=500,
                    help='umap number of epochs')

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
                    transform=transforms.Compose([PadToSquare(), transforms.Resize([224, 224]), transforms.ToTensor()]))

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
        features = np.load("{}/features.npy".format(args.features_folder))
        targets = np.load("{}/targets.npy".format(args.features_folder))

    # exit()

    hyper_params = "n_neighbors_{}_min_dist_{}_n_epochs_{}".format(args.n_neighbors, args.min_dist, args.n_epochs)
    # calculating UMAP
    if not os.path.isfile("{}/umap_{}.npy".format(args.save_folder, hyper_params)):
        print("Calculating UMAP")
        reducer = cuml.UMAP(
                            n_neighbors=args.n_neighbors,
                            n_components=args.n_components,
                            min_dist=args.min_dist,
                            n_epochs=args.n_epochs
                        ).fit(features)
        
        # umap = reducer.fit_transform(features)
        # reducer = umap.UMAP(n_neighbors=20,
        #                     n_components=args.n_components,
        #                     min_dist=0.0).fit(features)
        embedding = reducer.transform(features)
        np.save("{}/umap_{}.npy".format(args.save_folder, hyper_params), embedding)
    else:
        print("Loading UMAP")
        embedding = np.load("{}/umap_{}.npy".format(args.save_folder, hyper_params))

    # plot only 2d results
    if args.n_components == 2:
        print("Plotting UMAP")
        
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
        fig.savefig('{}/UMAP_{}.svg'.format(args.save_folder, hyper_params))
        # legendFig.savefig('{}/UMAP_legend.svg'.format(args.save_folder))

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
        fig.savefig('{}/UMAP_2x_{}.svg'.format(args.save_folder, hyper_params))
        plt.close()

if __name__ == '__main__':
    main()
