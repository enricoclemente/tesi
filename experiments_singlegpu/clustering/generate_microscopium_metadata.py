#!/usr/bin/env python
from cProfile import label
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
import open_clip
from experiments_singlegpu.datasets.SocialProfilePictures import SocialProfilePictures

import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import cuml
from sklearn.mixture import GaussianMixture
from SPICE.spice.config import Config
from sklearn.manifold import TSNE
from experiments_singlegpu.clustering.utils import calculate_clustering_accuracy_expanded, calculate_clustering_accuracy_expanded_with_overclustering, calculate_acc_overclustering
import matplotlib.lines as mlines
from sklearn.metrics import silhouette_samples, silhouette_score, normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
import matplotlib.cm as cm
import csv



parser = argparse.ArgumentParser(description='Generate microscopium csv')
parser.add_argument("--config_file", default="./experiments_config_example.py", metavar="FILE",
                    help="path to config file (same used with compute UMAP+GMM)", type=str)
parser.add_argument('--dataset_folder', metavar='DIR', default='./datasets/cifar10',
                    help='path to dataset')
parser.add_argument('--cluster_predictions', metavar="FILE",
                    help="path to cluster predictions .npy file" )
parser.add_argument('--n_clusters', type=int)
parser.add_argument('--n_neighbors', type=int)
parser.add_argument('--min_dist', type=float)
parser.add_argument('--save_folder', metavar='DIR', default='./results',
                    help='path where to save results')


def main():
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_file)

    # setting of save_folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    train_dataset = SocialProfilePictures(version=cfg.dataset.version, root=args.dataset_folder, split='train', randomize_metadata=cfg.dataset.randomize_metadata,
                    transform=transforms.Compose([  transforms.Resize([cfg.dataset.img_size, cfg.dataset.img_size]), 
                                                    transforms.ToTensor()]))
    test_dataset = SocialProfilePictures(version=cfg.dataset.version, root=args.dataset_folder, split='test', randomize_metadata=cfg.dataset.randomize_metadata,
                    transform=transforms.Compose([  transforms.Resize([cfg.dataset.img_size, cfg.dataset.img_size]), 
                                                    transforms.ToTensor()]))

    if cfg.model == "resnet18":
        model = resnet18()
        if not os.path.isfile(cfg.model_path):
            # if not available a previously trained model on moco use pretrained one on Imagenet
            print("Using pretrained model on ImageNet")
            model = resnet18(pretrained=True)
        else:
            model = resnet18(num_classes=128)
            loc = 'cuda:{}'.format(torch.cuda.current_device())
            checkpoint = torch.load(cfg.model_path, map_location=loc)
            dim_mlp = model.fc.in_features
            model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
            state_dict = dict()
            for key in checkpoint['state_dict']:
                
                if key.startswith("encoder_q"):
                    # print(key[22:])
                    state_dict[key[10:]] = checkpoint['state_dict'][key]
            model.load_state_dict(state_dict, strict=False)
        
        if cfg.features_layer == 'layer4':
            # removing fc layer 
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
    elif cfg.model == "vit_b_32":
        if not os.path.isfile(cfg.model_path):
            # if not available a previously trained model on moco use pretrained one on Imagenet
            print("Using pretrained model on LAION-2B E16")
            model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_e16')
        else:
            raise NotImplementedError("Not yet implemented trained ViT model")
    else:
        raise NotImplementedError("Choose a valid model!")
    
    for p in model.parameters():
        p.requires_grad = False

    # print(model)
    model.cuda()

    print("Extracting train features")
    train_features, train_targets = extract_features_targets(model, train_dataset, cfg)

    print("Extracting test features")
    test_features, test_targets = extract_features_targets(model, test_dataset, cfg)

    print("Calculating t-SNE")
    tsne_2d = TSNE(n_components=2).fit_transform(test_features)
    tx = tsne_2d[:, 0]
    ty = tsne_2d[:, 1]
    
    print("Calculating UMAP with")
    umap = cuml.UMAP(   n_components=args.n_clusters,
                        n_neighbors=args.n_neighbors,
                        min_dist=args.min_dist,
                        n_epochs=cfg.umap.n_epochs
                    ).fit(train_features)
    train_embedding = umap.transform(train_features)
    test_embedding = umap.transform(test_features)

    print("Calculating GMM on UMAP embedding")
    gmm = GaussianMixture(n_components=8).fit(train_embedding)
    cluster_predictions = gmm.predict(test_embedding)
    
    # cluster_predictions = np.load(args.cluster_predictions)

    # this is needed beacause microscopium needs relative path to open sprites
    dataset_relative_path = os.path.relpath(args.dataset_folder, args.save_folder)

    with open(os.path.join(args.save_folder, 'clutering_metadata.csv'), mode='w') as file:
        csv_writer = csv.writer(file, delimiter=',')
        csv_writer.writerow(['img_name', 'img_folder', 'class', 'cluster_prediction', 'img_path', 'x.tsne', 'y.tsne'])
        for i, meta in enumerate(test_dataset.metadata):
            csv_writer.writerow([meta['img_name'], meta['img_folder'], meta['target']['target_level'], 'cluster ' + str(cluster_predictions[i]), os.path.join(dataset_relative_path, meta['img_folder'], meta['img_name']), tx[i], ty[i]])

    plot_GMM(tsne_2d, cluster_predictions, args.n_clusters, args.save_folder, "")

def extract_features_targets(model, dataset, cfg):
    features = []
    targets = []

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1, 
                pin_memory=True, drop_last=False)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(loader):
            img = img.cuda()

            if cfg.model == "resnet18":
                feature = model.forward(img)
            elif cfg.model == "vit_b_32":
                feature = model.encode_image(img)
            
            if cfg.features_layer == 'layer4' and cfg.model == "resnet18":
                # flattening and normalizing if extracting from non fc layers
                feature = torch.flatten(feature, 1)
            
            feature = F.normalize(feature, dim=1)

            # collecting all features and targets
            features.append(feature.cpu())
            targets.append(target)
            
            print("[{}]/[{}] batch iteration".format(i, len(loader)))

    features = torch.cat(features)
    features = features.numpy()

    targets = torch.cat(targets)
    targets = targets.numpy()

    return features, targets


def plot_tSNE(dataset, embedding, targets, save_folder):
    colors_per_class = {}
    for class_name in dataset.classes:
        colors_per_class[class_name] = list(np.random.choice(range(256), size=3))
    
    # extract x and y coordinates 
   

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
    fig.savefig('{}/t-SNE.svg'.format(save_folder))
    # legendFig.savefig('{}/UMAP_legend.svg'.format(args.save_folder))
    plt.close()


def plot_GMM(embedding, predicted_cluster_labels, num_clusters, save_folder, prefix):
    tx = embedding[:, 0]
    ty = embedding[:, 1]

    colors_per_cluster = []
    for cluster in range(num_clusters):
        colors_per_cluster.append(list(np.random.choice(range(256), size=3)))
    
    colored_clusters = []
    for i in range(len(embedding)):
        colored_clusters.append(np.array(colors_per_cluster[predicted_cluster_labels[i]], dtype=float)/255)
    fig = plt.figure("GMM", figsize=(20,15))
    ax = fig.add_subplot(111)
    ax.scatter(tx, ty, c=colored_clusters)
    fig.savefig('{}/{}GMM.svg'.format(save_folder, prefix + "_" if prefix else ""))
    plt.close()


if __name__ == '__main__':
    main()
