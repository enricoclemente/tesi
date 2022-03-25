#!/usr/bin/env python
import sys
import random
import os
import argparse
sys.path.insert(0, './')

import random
import numpy as np
import torch.nn.functional as F
import torch

from torchvision.models.resnet import resnet18
from experiments_singlegpu.datasets.SocialProfilePictures import SocialProfilePictures

import torchvision.transforms as transforms
from experiments_singlegpu.datasets.utils.custom_transforms import PadToSquare

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser(description='t-SNE calculator')
parser.add_argument('--save_folder', metavar='DIR', default='./results/cifar10/moco',
                    help='path to results')

def main():
    args = parser.parse_args()
    # set seed in order to reproduce always the same result
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = SocialProfilePictures(root='/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets', split=['train', 'validation', 'test'],
                    transform=transforms.Compose([PadToSquare(), transforms.Resize([224, 224]), transforms.ToTensor()]))

    if not (os.path.isfile("{}/features.npy".format(args.save_folder)) 
            and os.path.isfile("{}/targets.npy".format(args.save_folder))):
        print("No presaved features and targets found!")
        model = resnet18(pretrained=True)
        # removing fc layer
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


    # extracting and plotting 2d t-SNE
    colors_per_class = {}
    for class_name in dataset.classes:
        colors_per_class[class_name] = list(np.random.choice(range(256), size=3))
    
    if not os.path.isfile("{}/tsne_2d.npy".format(args.save_folder)):
        print("Calculating t-SNE")
        tsne_2d = TSNE(n_components=2).fit_transform(features)
        np.save("{}/tsne_2d.npy".format(args.save_folder), tsne_2d)
    else:
        print("Loading t-SNE")
        tsne_2d = np.load("{}/tsne_2d.npy".format(args.save_folder))


    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne_2d[:, 0]
    ty = tsne_2d[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)


    # initialize a matplotlib plot
    fig = plt.figure("t-SNE", figsize=(20,15))

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
    legendFig = plt.figure("t-SNE legend", figsize=(5,7))
    legendFig.legend(plots, dataset.classes, loc='center')

    # finally, show the plot
    fig.savefig('{}/t-SNE_2d.svg'.format(args.save_folder))
    legendFig.savefig('{}/t-SNE_legend.svg'.format(args.save_folder))

    plt.close()

    # plotting 2d t-SNE at double dims
    fig = plt.figure("t-SNE 2x", figsize=(40,30))

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
    fig.savefig('{}/t-SNE_2d_2x.svg'.format(args.save_folder))

    plt.close()

    exit()
    # extracting and plotting 3d t-SNE
    if not os.path.isfile("{}/tsne_3d.npy".format(args.save_folder)):
        print("Calculating t-SNE")
        tsne_3d = TSNE(n_components=3).fit_transform(features)
        np.save("{}/tsne_3d.npy".format(args.save_folder), tsne_3d)
    else:
        print("Loading t-SNE")
        tsne_3d = np.load("{}/tsne_3d.npy".format(args.save_folder))


    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne_3d[:, 0]
    ty = tsne_3d[:, 1]
    tz = tsne_3d[:, 2]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    tz = scale_to_01_range(tz)


    # initialize a matplotlib plot
    fig = plt.figure("t-SNE", figsize=(20,15))

    ax = fig.add_subplot(111, projection='3d')
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
        current_tz = np.take(tz, indices)

        # convert the class color to matplotlib format
        color = np.array(colors_per_class[class_name], dtype=float) / 255

        # add a scatter plot with the corresponding color and label
        plots[class_index] = ax.scatter(current_tx, current_ty, current_tz, c=color, label=class_name)
    
    ax.legend(bbox_to_anchor=(1.0, 1.0))
    # finally, show the plot
    fig.savefig('{}/t-SNE_3d.svg'.format(args.save_folder))


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
