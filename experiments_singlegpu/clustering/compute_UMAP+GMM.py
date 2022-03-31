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
from sklearn.mixture import GaussianMixture
from SPICE.spice.utils.evaluation import calculate_acc, calculate_nmi, calculate_ari


parser = argparse.ArgumentParser(description='UMAP+GMM calculator (experiments)')
parser.add_argument('--features_folder', metavar='DIR', default='./features',
                    help='path to previously calculated features')
parser.add_argument('--save_folder', metavar='DIR', default='./results',
                    help='path to results')
parser.add_argument('--umap_n_components', type=int, default=2,
                    help='umap number of components')
parser.add_argument('--umap_n_epochs', type=int, default=1000,
                    help='umap number of epochs to train')
parser.add_argument('--gmm_n_components', type=int, default=2,
                    help='gmm number of components')


def main():
    args = parser.parse_args()

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


    n_neighbors_values = [5, 10, 20, 50]
    min_dist_values = [0.0, 0.1, 0.25, 0.5]

    best_acc = 0
    best_acc_hyperparams = ""
    for n_neighbors in n_neighbors_values:
        for min_dist in min_dist_values:
            """
                Compute UMAP
            """
            umap_hyperparams = "n_components_{}_n_neighbors_{}_min_dist_{}_n_epochs_{}".format(args.umap_n_components,n_neighbors, min_dist, args.n_epochs)
            # calculating UMAP
            if not os.path.isfile("{}/umap_{}.npy".format(args.save_folder, umap_hyperparams)):
                print("Calculating UMAP with {}".format(umap_hyperparams))
                reducer = cuml.UMAP(
                                    n_neighbors=n_neighbors,
                                    n_components=args.umap_n_components,
                                    min_dist=min_dist,
                                    n_epochs=args.umap_n_epochs
                                ).fit(features)
                embedding = reducer.transform(features)
                np.save("{}/umap_{}.npy".format(args.save_folder, umap_hyperparams), embedding)
            else:
                print("Loading UMAP with {}".format(umap_hyperparams))
                embedding = np.load("{}/umap_{}.npy".format(args.save_folder, umap_hyperparams))
            
            if args.umap_n_components != 2:
                if not os.path.isfile("{}/umap_{}.npy".format(args.save_folder, umap_hyperparams)):
                    print("Calculating 2d UMAP with {}".format(umap_hyperparams))
                    reducer_2d = cuml.UMAP(
                                        n_neighbors=n_neighbors,
                                        n_components=args.umap_n_components,
                                        min_dist=min_dist,
                                        n_epochs=args.umap_n_epochs
                                    ).fit(features)
                    embedding_2d = reducer_2d.transform(features)
                    np.save("{}/umap_2d_{}.npy".format(args.save_folder, umap_hyperparams), embedding_2d)
                else:
                    print("Loading UMAP with {}".format(umap_hyperparams))
                    embedding_2d = np.load("{}/umap_2d_{}.npy".format(args.save_folder, umap_hyperparams)) 
            else:
                embedding_2d = embedding
        
            print("Plotting 2d UMAP")
            colors_per_class = {}
            for class_name in dataset.classes:
                colors_per_class[class_name] = list(np.random.choice(range(256), size=3))
            
            # extract x and y coordinates 
            tx = embedding_2d[:, 0]
            ty = embedding_2d[:, 1]

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
            fig.savefig('{}/UMAP_{}.svg'.format(args.save_folder, umap_hyperparams))
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
            # build a legend using the labels we set previously
            ax.legend(bbox_to_anchor=(1.0, 1.0))

            # finally, show the plot
            fig.savefig('{}/UMAP_2x_{}.svg'.format(args.save_folder, umap_hyperparams))
            plt.close()

            """
                Compute GMM starting from previously calculated embeddings with UMAP
            """
            gmm_hyperparams = "n_components_{}".format(args.gmm_n_components)
            if not os.path.isfile("{}/gmm_{}_umap_{}.npy".format(args.save_folder, gmm_hyperparams, umap_hyperparams)):
                print("Calculating GMM with {} on UMAP embedding with {}".format(umap_hyperparams, gmm_hyperparams))
                gmm = GaussianMixture(n_components=args.n_components).fit(embedding)
                predicted_cluster_labels = gmm.predict(embedding)
                np.save("{}/gmm_{}_umap_{}.npy".format(args.save_folder, gmm_hyperparams, umap_hyperparams), predicted_cluster_labels)
            else:
                print("Loading GMM with {} on UMAP embedding with {}".format(umap_hyperparams, gmm_hyperparams))
                predicted_cluster_labels = np.load("{}/gmm_{}_umap_{}.npy".format(args.save_folder, gmm_hyperparams, umap_hyperparams)) 

            gt_labels = dataset.targets
            acc = calculate_acc(predicted_cluster_labels, gt_labels)
            nmi = calculate_nmi(predicted_cluster_labels, gt_labels)
            ari = calculate_ari(predicted_cluster_labels, gt_labels)

            print("Plotting GMM")
            colors_per_cluster = []
            for cluster in range(args.n_components):
                colors_per_cluster.append(list(np.random.choice(range(256), size=3)))
            
            colored_clusters = []
            for i in range(len(embedding)):
                colored_clusters.append(np.array(colors_per_cluster[predicted_cluster_labels[i]], dtype=float)/255)
            fig = plt.figure("GMM", figsize=(20,15))
            ax = fig.add_subplot(111)
            ax.scatter(tx, ty, c=colored_clusters, s=5)
            fig.savefig('{}/GMM_{}_UMAP_{}.svg'.format(args.save_folder, gmm_hyperparams, umap_hyperparams))
            plt.close()

            # plotting at double dims
            fig = plt.figure("GMM 2x", figsize=(40,30))
            ax = fig.add_subplot(111)
            ax.scatter(tx, ty, c=colored_clusters, s=5)
            fig.savefig('{}/GMM_2x_{}_UMAP_{}.svg'.format(args.save_folder, gmm_hyperparams, umap_hyperparams))
            plt.close()

            print("UMAP embedding with n_components={}, n_neighbors={}, min_dist={}".format(args.umap_n_components, n_neighbors, min_dist))
            print("\tGMM scores: ACC: {} NMI: {} ARI: {}".format(acc, nmi, ari))

            with open("{}/results.txt".format(args.save_folder), 'a') as file:
                file.write("UMAP embedding with n_components={}, n_neighbors={}, min_dist={}".format(args.umap_n_components, n_neighbors, min_dist))
                file.write("\tGMM scores: ACC: {} NMI: {} ARI: {}".format(acc, nmi, ari))
            
            if acc > best_acc:
                best_acc = acc
                best_acc_hyperparams = "UMAP with {} GMM with {}".format(umap_hyperparams, gmm_hyperparams)
    
    with open("{}/results.txt".format(args.save_folder), 'a') as file:
        file.write("Best ACC: {} obtained calculating {}".format(best_acc, best_acc_hyperparams))
            
if __name__ == '__main__':
    main()
