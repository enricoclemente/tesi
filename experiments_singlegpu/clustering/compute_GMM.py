#!/usr/bin/env python
import sys
import random
import os
import argparse
sys.path.insert(0, './')

import random
import numpy as np

from experiments_singlegpu.datasets.SocialProfilePictures import SocialProfilePictures

import torchvision.transforms as transforms
from experiments_singlegpu.datasets.utils.custom_transforms import PadToSquare

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from SPICE.spice.utils.evaluation import calculate_acc, calculate_nmi, calculate_ari


parser = argparse.ArgumentParser(description='GMM calculator')
parser.add_argument('--embedding', type=str, metavar='PATH', default='./results/umap/umap.npy',
                    help='path to the previously calculated manifold')
parser.add_argument('--save_folder', metavar='DIR', default='./results/cifar10/moco',
                    help='path to results')
parser.add_argument('--n_components', type=int, default=2,
                    help='umap number of components')

def main():
    args = parser.parse_args()

    # setting of save_folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    if not os.path.exists(args.embedding):
        raise NotImplementedError("You must first compute embeddings")
    
    print("Loading UMAP")
    embedding = np.load("{}".format(args.embedding))
    
    gmm = GaussianMixture(n_components=args.n_components).fit(embedding)
    predicted_cluster_labels = gmm.predict(embedding)

    # calcolo ACC, NMI, ARI
    dataset = SocialProfilePictures(root='/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets', split=['train', 'val', 'test'],
                    transform=transforms.Compose([PadToSquare(), transforms.Resize([225, 225]), transforms.ToTensor()]))

    gt_labels = dataset.targets
    acc = calculate_acc(predicted_cluster_labels, gt_labels)
    nmi = calculate_nmi(predicted_cluster_labels, gt_labels)
    ari = calculate_ari(predicted_cluster_labels, gt_labels)

    print("GMM scores: ACC: {} NMI: {} ARI: {}".format(acc, nmi, ari))

    # plotting

    colors_per_cluster = []
    for cluster in range(args.n_components):
        colors_per_cluster.append(list(np.random.choice(range(256), size=3)))
    
    colored_clusters = []
    for i in range(len(embedding)):
        colored_clusters.append(np.array(colors_per_cluster[predicted_cluster_labels[i]], dtype=float)/255)
    fig = plt.figure("GMM", figsize=(20,15))
    ax = fig.add_subplot(111)
    ax.scatter(embedding[:, 0], embedding[:, 1], c=colored_clusters, s=5)
    fig.savefig('{}/GMM.svg'.format(args.save_folder))
    plt.close()

    # plotting at double dims
    fig = plt.figure("GMM 2x", figsize=(40,30))
    ax = fig.add_subplot(111)
    ax.scatter(embedding[:, 0], embedding[:, 1], c=colored_clusters, s=5)
    fig.savefig('{}/GMM_2x.svg'.format(args.save_folder))
    plt.close()

if __name__ == '__main__':
    main()
