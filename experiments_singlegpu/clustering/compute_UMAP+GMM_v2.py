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

import matplotlib.pyplot as plt
import cuml
from sklearn.mixture import GaussianMixture
from SPICE.spice.utils.evaluation import calculate_acc, calculate_nmi, calculate_ari
from SPICE.spice.config import Config
from sklearn.manifold import TSNE
from experiments_singlegpu.clustering.utils import calculate_clustering_accuracy_expanded
import matplotlib.lines as mlines


parser = argparse.ArgumentParser(description='UMAP+GMM calculator (experiments)')
parser.add_argument("--config_file", default="./experiments_config_example.py", metavar="FILE",
                    help="path to config file (same used with moco training)", type=str)
parser.add_argument('--dataset_folder', metavar='DIR', default='./datasets/cifar10',
                    help='path to dataset')
parser.add_argument('--model_path', type=str, default='./results/checkpoint_last.pth.tar',
                    help='The pretrained model path')
parser.add_argument('--save_folder', metavar='DIR', default='./results',
                    help='path to results')
parser.add_argument('--features_layer', type=str, default='layer4',
                    help='layer from which to extract features')
parser.add_argument('--umap_n_epochs', type=int, default=1000,
                    help='umap number of epochs to train')


def main():
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_file)

    # set seed in order to reproduce always the same result
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # setting of save_folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    train_dataset = SocialProfilePictures(version=cfg.dataset.version, root=args.dataset_folder, split='train', randomize_metadata=cfg.dataset.randomize_metadata,
                    transform=transforms.Compose([  transforms.Resize([cfg.dataset.img_size, cfg.dataset.img_size]), 
                                                    transforms.ToTensor()]))
    val_dataset = SocialProfilePictures(version=cfg.dataset.version, root=args.dataset_folder, split='val', randomize_metadata=cfg.dataset.randomize_metadata,
                    transform=transforms.Compose([  transforms.Resize([cfg.dataset.img_size, cfg.dataset.img_size]), 
                                                    transforms.ToTensor()]))
    test_dataset = SocialProfilePictures(version=cfg.dataset.version, root=args.dataset_folder, split='test', randomize_metadata=cfg.dataset.randomize_metadata,
                    transform=transforms.Compose([  transforms.Resize([cfg.dataset.img_size, cfg.dataset.img_size]), 
                                                    transforms.ToTensor()]))

    train_dataset = test_dataset

    model = resnet18()
    if not os.path.isfile(args.model_path):
        # if not available a previously trained model on moco use pretrained one on Imagenet
        model = resnet18(pretrained=True)
    else:
        model = resnet18(num_classes=128)
        loc = 'cuda:{}'.format(torch.cuda.current_device())
        checkpoint = torch.load(args.model_path, map_location=loc)
        # 
        dim_mlp = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
        state_dict = dict()
        for key in checkpoint['state_dict']:
            print(key)
            if key.startswith("encoder_q"):
                # print(key[22:])
                state_dict[key[10:]] = checkpoint['state_dict'][key]
        model.load_state_dict(state_dict, strict=False)
    
    
    if args.features_layer == 'layer4':
        # removing fc layer 
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
    
    for p in model.parameters():
        p.requires_grad = False

    print(model)
    model.cuda()

    print("Extracting train features")
    train_features, train_targets = extract_features_targets(model, train_dataset, args)

    # print("Extracting validation features")
    # val_features, val_targets = extract_features_targets(model, val_dataset, args)

    # print("Extracting test features")
    # test_features, test_targets = extract_features_targets(model, test_dataset, args)

    test_features = train_features
    val_features = test_features
    test_targets = train_targets
    val_targets = test_targets

    best_acc = {}
    best_nmi = {}
    best_ari = {}
    
    print("Calculating t-SNE")
    num_clusters = len(train_dataset.classes)
    tsne_2d = TSNE(n_components=2).fit_transform(test_features)
    num_clusters_values = [train_dataset.classes, 2 * train_dataset.classes, 3 * train_dataset.classes]
    n_neighbors_values = [5, 10, 20, 50]
    min_dist_values = [0.0, 0.1, 0.25, 0.5]
    for n_neighbors in n_neighbors_values:
        for min_dist in min_dist_values:
            """
                Compute UMAP
            """
            umap_hyperparams = "n_components_{}_n_neighbors_{}_min_dist_{}_n_epochs_{}".format(num_clusters,n_neighbors, min_dist, args.umap_n_epochs)
            gmm_hyperparams = "n_components_{}".format(num_clusters)
            
            # calculating UMAP
            print("Calculating UMAP with {}".format(umap_hyperparams))
            umap = cuml.UMAP(
                                n_neighbors=n_neighbors,
                                n_components=num_clusters,
                                min_dist=min_dist,
                                n_epochs=args.umap_n_epochs
                            ).fit(train_features)
            train_embedding = umap.transform(train_features)
            val_embedding = umap.transform(val_features)
            """
                Compute GMM starting from previously calculated embeddings with UMAP
            """
            print("Calculating GMM with {} on UMAP embedding with {}".format(gmm_hyperparams, umap_hyperparams))
            gmm = GaussianMixture(n_components=num_clusters).fit(train_embedding)
            predicted_cluster_labels = gmm.predict(val_embedding)
            
            try:
                acc = calculate_acc(predicted_cluster_labels, val_targets)
            except AssertionError as msg:
                print("Clustering Accuracy failed: ") 
                print(msg)
                acc = -1
            nmi = calculate_nmi(predicted_cluster_labels, val_targets)
            ari = calculate_ari(predicted_cluster_labels, val_targets)

            print("UMAP embedding with n_components={}, n_neighbors={}, min_dist={}".format(num_clusters, n_neighbors, min_dist))
            print("\tGMM scores: ACC: {} NMI: {} ARI: {}".format(acc, nmi, ari))

            with open("{}/hyperpameters_search.txt".format(args.save_folder), 'a') as file:
                file.write("UMAP embedding with n_components={}, n_neighbors={}, min_dist={}\n".format(num_clusters, n_neighbors, min_dist))
                file.write("\tACC: {} NMI: {} ARI: {}\n".format(acc, nmi, ari))
            
            if acc > best_acc['acc']:
                best_acc['acc'] = acc
                best_acc['nmi'] = nmi
                best_acc['ari'] = ari
                best_acc['umap'] = umap
                best_acc['gmm'] = gmm
                best_acc["umap_hyperparams"] = umap_hyperparams
                best_acc["gmm_hyperparams"] = gmm_hyperparams
                best_acc["hyperparams"] = "GMM with {} UMAP with {}".format(gmm_hyperparams, umap_hyperparams)
            
            if nmi > best_nmi['nmi']:
                best_nmi['acc'] = acc
                best_nmi['nmi'] = nmi
                best_nmi['ari'] = ari
                best_nmi['umap'] = umap
                best_nmi['gmm'] = gmm
                best_nmi["umap_hyperparams"] = umap_hyperparams
                best_nmi["gmm_hyperparams"] = gmm_hyperparams
                best_nmi["hyperparams"] = "GMM with {} UMAP with {}".format(gmm_hyperparams, umap_hyperparams)
            
            if ari > best_ari['ari']:
                best_ari['acc'] = acc
                best_ari['nmi'] = nmi
                best_ari['ari'] = ari
                best_ari['umap'] = umap
                best_ari['gmm'] = gmm
                best_ari["umap_hyperparams"] = umap_hyperparams
                best_ari["gmm_hyperparams"] = gmm_hyperparams
                best_ari["hyperparams"] = "GMM with {} UMAP with {}".format(gmm_hyperparams, umap_hyperparams)
    
    print("Testing {}".format(best_acc["hyperparams"]))
    test_embedding = best_acc['umap'].transform(test_features)
    best_acc['clustering_predictions'] = best_acc['gmm'].predict(test_embedding)

    best_acc["acc"], best_acc['acc_per_class_total'], best_acc['acc_per_class_relative'], best_acc['cluster_class_assigned'] = calculate_clustering_accuracy_expanded(best_acc['clustering_predictions'], test_targets, len(test_dataset.classes))
    best_acc["nmi"] = calculate_nmi(best_acc['clustering_predictions'], test_targets)
    best_acc["ari"] = calculate_ari(best_acc['clustering_predictions'], test_targets)

    print("Testing {}".format(best_nmi["hyperparams"]))
    test_embedding = best_nmi['umap'].transform(test_features)
    best_nmi['clustering_predictions'] = best_nmi['gmm'].predict(test_embedding)

    best_nmi["acc"], best_nmi['acc_per_class_total'], best_nmi['acc_per_class_relative'], best_nmi['cluster_class_assigned'] = calculate_clustering_accuracy_expanded(best_nmi['clustering_predictions'], test_targets, len(test_dataset.classes))
    best_nmi["nmi"] = calculate_nmi(best_nmi['clustering_predictions'], test_targets)
    best_nmi["ari"] = calculate_ari(best_nmi['clustering_predictions'], test_targets)
    
    print("Testing {}".format(best_ari["hyperparams"]))
    test_embedding = best_ari['umap'].transform(test_features)
    best_ari['clustering_predictions'] = best_ari['gmm'].predict(test_embedding)

    best_ari["acc"], best_ari['acc_per_class_total'], best_ari['acc_per_class_relative'], best_ari['cluster_class_assigned'] = calculate_clustering_accuracy_expanded(best_ari['clustering_predictions'], test_targets, len(test_dataset.classes))
    best_ari["nmi"] = calculate_nmi(best_ari['clustering_predictions'], test_targets)
    best_ari["ari"] = calculate_ari(best_ari['clustering_predictions'], test_targets)
    

    with open("{}/results.txt".format(args.save_folder), 'a') as file:
        plot_tSNE(test_dataset, tsne_2d, test_targets, args.save_folder, "", "")
        
        f = lambda v: np.round(v, 2)
        np.save("{}/best_acc_cluster_predictions_{}.npy".format(args.save_folder,  best_acc["hyperparams"]), best_acc["clustering_predictions"])
        file.write("Best ACC obtained calculating {}\n\tACC: {} NMI: {} ARI: {}\n".format(best_acc["hyperparams"], best_acc['acc'], best_acc['nmi'], best_acc['ari']))
        # plotting best acc umap and gmm
        plot_GMM(tsne_2d, best_acc["clustering_predictions"], args.save_folder, best_acc["umap_hyperparams"], best_acc["gmm_hyperparams"], num_clusters, "best_acc")
        plot_GMM_with_class_assigned(tsne_2d, best_acc["clustering_predictions"], best_acc['cluster_class_assigned'], test_dataset.classes, args.save_folder, best_acc["umap_hyperparams"], best_acc["gmm_hyperparams"], num_clusters, "best_acc")
        plot_bar_chart(test_dataset.classes, f(best_acc['acc_per_class_total']), 'Best ACC accuracy per class total', 'accuracy', args.save_folder)
        plot_bar_chart(test_dataset.classes, f(best_acc['acc_per_class_relative']), 'Best ACC accuracy per class', 'accuracy', args.save_folder)
        
        np.save("{}/best_nmi_cluster_predictions_{}.npy".format(args.save_folder,  best_nmi["hyperparams"]), best_nmi["clustering_predictions"])
        file.write("Best NMI obtained calculating {}\n\tACC: {} NMI: {} ARI: {}\n".format(best_nmi["hyperparams"], best_nmi['acc'], best_nmi['nmi'], best_nmi['ari']))
        # plotting best nmi umap and gmm
        plot_GMM(tsne_2d, best_nmi["clustering_predictions"], args.save_folder, best_nmi["umap_hyperparams"], best_nmi["gmm_hyperparams"], num_clusters, "best_nmi")
        plot_GMM_with_class_assigned(tsne_2d, best_nmi["clustering_predictions"], best_nmi['cluster_class_assigned'], test_dataset.classes, args.save_folder, best_nmi["umap_hyperparams"], best_nmi["gmm_hyperparams"], num_clusters, "best_nmi")
        plot_bar_chart(test_dataset.classes, f(best_nmi['acc_per_class_total']), 'Best NMI accuracy per class total', 'accuracy', args.save_folder)
        plot_bar_chart(test_dataset.classes, f(best_nmi['acc_per_class_relative']), 'Best NMI accuracy per class', 'accuracy', args.save_folder)

        np.save("{}/best_ari_cluster_predictions_{}.npy".format(args.save_folder,  best_ari["hyperparams"]), best_ari["clustering_predictions"])
        file.write("Best ARI obtained calculating {}\n\tACC: {} NMI: {} ARI: {}\n".format(best_ari["hyperparams"], best_ari['acc'], best_ari['nmi'], best_ari['ari']))
        # plotting best nmi umap and gmm
        plot_GMM(tsne_2d, best_ari["clustering_predictions"], args.save_folder, best_ari["umap_hyperparams"], best_ari["gmm_hyperparams"], num_clusters, "best_ari")
        plot_GMM_with_class_assigned(tsne_2d, best_ari["clustering_predictions"], best_ari['cluster_class_assigned'], test_dataset.classes, args.save_folder, best_ari["umap_hyperparams"], best_ari["gmm_hyperparams"], num_clusters, "best_ari")       
        plot_bar_chart(test_dataset.classes, f(best_ari['acc_per_class_total']), 'Best ARI accuracy per class total', 'accuracy', args.save_folder)
        plot_bar_chart(test_dataset.classes, f(best_ari['acc_per_class_relative']), 'Best ARI accuracy per class', 'accuracy', args.save_folder)



def extract_features_targets(model, dataset, args):
    features = []
    targets = []

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1, 
                pin_memory=True, drop_last=False)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(loader):
            img = img.cuda()

            feature = model.forward(img)
            if args.features_layer == 'layer4':
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


def plot_tSNE(dataset, embedding, targets, save_folder, umap_hyperparams, prefix):
    colors_per_class = {}
    for class_name in dataset.classes:
        colors_per_class[class_name] = list(np.random.choice(range(256), size=3))
    
    # extract x and y coordinates 
    tx = embedding[:, 0]
    ty = embedding[:, 1]

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
    fig.savefig('{}/{}_t-SNE.svg'.format(save_folder, prefix, umap_hyperparams))
    # legendFig.savefig('{}/UMAP_legend.svg'.format(args.save_folder))
    plt.close()


def plot_UMAP(dataset, embedding, targets, save_folder, umap_hyperparams, prefix):
    colors_per_class = {}
    for class_name in dataset.classes:
        colors_per_class[class_name] = list(np.random.choice(range(256), size=3))
    
    # extract x and y coordinates 
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
    fig.savefig('{}/{}_UMAP_{}.svg'.format(save_folder, prefix, umap_hyperparams))
    # legendFig.savefig('{}/UMAP_legend.svg'.format(args.save_folder))
    plt.close()


def plot_GMM(embedding, predicted_cluster_labels, save_folder, umap_hyperparams, gmm_hyperparams, num_clusters, prefix):
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
    fig.savefig('{}/{}_GMM_{}_UMAP_{}.svg'.format(save_folder, prefix, gmm_hyperparams, umap_hyperparams))
    plt.close()


def plot_GMM_with_class_assigned(embedding, predicted_cluster_labels, cluster_class_assigned, classes_names, save_folder, umap_hyperparams, gmm_hyperparams, num_clusters, prefix):
    tx = embedding[:, 0]
    ty = embedding[:, 1]

    colors_per_cluster = []
    for cluster in range(num_clusters):
        colors_per_cluster.append(list(np.random.choice(range(256), size=3)))
    
    colored_clusters = []
    for i in range(len(embedding)):
        colored_clusters.append(np.array(colors_per_cluster[predicted_cluster_labels[i]], dtype=float)/255)
    fig = plt.figure("GMM", figsize=(22,15))
    ax = fig.add_subplot(111)
    ax.scatter(tx, ty, c=colored_clusters)

    
    legend_elements = []
    legend_labels = []
    for i, c in enumerate(cluster_class_assigned):
        legend_item = mlines.Line2D([], [], color=np.array(colors_per_cluster[c], dtype=float)/255, marker='o', linestyle='None',
                          markersize=10)
        legend_elements.append(legend_item)
        legend_labels.append('cluster {} -> class {}'.format(c, classes_names[i]))
    ax.legend(legend_elements, legend_labels, bbox_to_anchor=(1.0, 1.0))
    fig.savefig('{}/{}_GMM_{}_UMAP_{}_with_class_assigned.svg'.format(save_folder, prefix, gmm_hyperparams, umap_hyperparams))
    plt.close()


def plot_bar_chart(x, y, title, ylabel, save_folder):
    plt.figure("Bar Chart", figsize=(12,6))
    plt.xticks(rotation=45, ha='right')
    rect = plt.bar(x, y)
    plt.bar_label(rect, padding=3)

    plt.gca().set(title=title, ylabel=ylabel)
    plt.tight_layout()

    bottom, top = plt.ylim()
    plt.ylim([bottom, top + top*0.2])
    
    plt.savefig("{}/{}.svg".format(save_folder, title.lower().replace(" ", "_")))
    plt.close()


if __name__ == '__main__':
    main()
