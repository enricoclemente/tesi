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
import time



parser = argparse.ArgumentParser(description='UMAP+GMM calculator (experiments)')
parser.add_argument("--config_file", default="./experiments_config_example.py", metavar="FILE",
                    help="path to config file (same used with moco training)", type=str)
parser.add_argument('--dataset_folder', metavar='DIR', default='./datasets/cifar10',
                    help='path to dataset')
parser.add_argument('--save_folder', metavar='DIR', default='./results',
                    help='path to results')


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
    
    for p in model.parameters():
        p.requires_grad = False

    # print(model)
    model.cuda()

    print("Extracting train features")
    train_features, train_targets = extract_features_targets(model, train_dataset, cfg)

    print("Extracting validation features")
    val_features, val_targets = extract_features_targets(model, val_dataset, cfg)

    print("Extracting test features")
    test_features, test_targets = extract_features_targets(model, test_dataset, cfg)

    """
        Following commented code to use only for tests
    """
    # train_features = np.load('/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/version_03/clustering/umap+gmm/exp_prova/features.npy')
    # train_targets = np.load('/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/version_03/clustering/umap+gmm/exp_prova/targets.npy')
    # test_features = train_features
    # val_features = test_features
    # test_targets = train_targets
    # val_targets = test_targets

    best_ext_scores = []
    best_int_scores = []

    print("Calculating t-SNE")
    tsne_2d = TSNE(n_components=2).fit_transform(test_features)
    plot_tSNE(test_dataset, tsne_2d, test_targets, args.save_folder)

    num_classes = len(train_dataset.classes)
    n_clusters_values = np.arange(cfg.n_cluster_values.start, cfg.n_cluster_values.end+2,2)
    n_neighbors_values = [5, 10, 20, 50]
    min_dist_values = [0.0, 0.1, 0.25, 0.5]
    original_save_folder = "" + args.save_folder
    for n_clusters in n_clusters_values:
        best_ext = {"ami": 0.0}
        best_int = {"silhouette_avg": 0.0}
        
        args.save_folder = os.path.join(original_save_folder, "n_clusters_{}".format(n_clusters))

        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

        for n_neighbors in n_neighbors_values:
            for min_dist in min_dist_values:
                """
                    Compute UMAP
                """
                umap_n_components = num_classes if cfg.umap.fix_n_components else n_clusters
                hyperparams_text = "umap_n_components_{}_n_neighbors_{}_min_dist_{}".format(umap_n_components, n_neighbors, min_dist)
                
                # calculating UMAP
                print("Calculating UMAP with {}".format(hyperparams_text))
                umap = cuml.UMAP(   n_components=umap_n_components,
                                    n_neighbors=n_neighbors,
                                    min_dist=min_dist,
                                    n_epochs=cfg.umap.n_epochs
                                ).fit(train_features)
                train_embedding = umap.transform(train_features)
                val_embedding = umap.transform(val_features)

                # if in the embeddings there are NaN values GMM fails
                if not (np.any(np.isnan(train_embedding)) or np.any(np.isnan(val_embedding))):
                    """
                        Compute GMM starting from previously calculated embeddings with UMAP
                    """
                    print("Calculating GMM on UMAP embedding with {}".format(hyperparams_text))
                    gmm = GaussianMixture(n_components=n_clusters).fit(train_embedding)
                    val_predicted_cluster_labels = gmm.predict(val_embedding)
                    
                    nmi = normalized_mutual_info_score(val_predicted_cluster_labels, val_targets)
                    ami = adjusted_mutual_info_score(val_predicted_cluster_labels, val_targets)
                    ari = adjusted_rand_score(val_predicted_cluster_labels, val_targets)
                    silhouette_avg = silhouette_score(val_features, val_predicted_cluster_labels)
                    
                    print("Clustering with n_cluster={}, n_neighbors={}, min_dist={}".format(n_clusters, n_neighbors, min_dist))
                    print("\tNMI: {} AMI: {} ARI: {}".format(round(nmi, 3), round(ami, 3), round(ari, 3)))
                    print("\tsilhouette_score: {}".format(round(silhouette_avg, 3)))

                    with open("{}/hyperpameters_search.txt".format(args.save_folder), 'a') as file:
                        file.write("NMI: {} AMI: {} ARI: {} silhouette: {}\tn_components={} n_neighbors={}, min_dist={}\n".format(round(nmi, 3), round(ami, 3), round(ari, 3), round(silhouette_avg, 3),umap_n_components, n_neighbors, min_dist))

                    if ami > best_ext['ami']:
                        best_ext['ami'] = ami
                        best_ext['umap'] = umap
                        best_ext['gmm'] = gmm
                        best_ext["hyperparams"] = hyperparams_text
                    
                    if silhouette_avg > best_int['silhouette_avg']:
                        best_int["silhouette_avg"] = silhouette_avg
                        best_int['umap'] = umap
                        best_int['gmm'] = gmm
                        best_int["hyperparams"] = hyperparams_text
                else:
                    print("Clustering with n_cluster={}, n_neighbors={}, min_dist={}".format(n_clusters, n_neighbors, min_dist))
                    print("Failed because UMAP returned embeddings with Infinite or NaN values")
                    with open("{}/hyperpameters_search.txt".format(args.save_folder), 'a') as file:
                        file.write("NMI: -1 AMI: -1 ARI: -1 silhouette: -1\tn_neighbors={}, min_dist={}\n".format(n_neighbors, min_dist))
        
        start = time.time()
        best_ext_score = {}

        print("Testing clustering with {}".format(best_ext["hyperparams"]))
        test_embedding = best_ext['umap'].transform(test_features)
        best_ext_score['clustering_predictions'] = best_ext['gmm'].predict(test_embedding)

        if cfg.calculate_acc:
            best_ext_score["acc"], best_ext_score['acc_per_class_total'], best_ext_score['acc_per_class_relative'], best_ext_score['cluster_class_assigned'] = calculate_clustering_accuracy_expanded_with_overclustering(best_ext_score['clustering_predictions'], test_targets, num_classes)
            f = lambda v: np.round(v, 2)
            plot_acc_per_class(test_dataset.classes, f(best_ext_score['acc_per_class_total']), 'Best ACC accuracy per class total', 'total', args.save_folder)
            plot_acc_per_class(test_dataset.classes, f(best_ext_score['acc_per_class_relative']), 'Best ACC accuracy per class relative', 'relative', args.save_folder)
        else:
            best_ext_score['acc'] = -1
        best_ext_score["nmi"] = normalized_mutual_info_score(best_ext_score['clustering_predictions'], test_targets)
        best_ext_score["ami"] = adjusted_mutual_info_score(best_ext_score['clustering_predictions'], test_targets)
        best_ext_score["ari"] = adjusted_rand_score(best_ext_score['clustering_predictions'], test_targets)
        best_ext_score["silhouette_avg"] = silhouette_score(test_features, best_ext_score['clustering_predictions'])
        best_ext_score["sample_silhouette_values"] = silhouette_samples(test_features, best_ext_score['clustering_predictions'])

        print("\nBest external clustering with n_clusters={}".format(n_clusters))
        print("\tACC: {} NMI: {} AMI: {} ARI: {} silhouette: {}\t{}\n".format(best_ext_score["acc"], best_ext_score["nmi"], best_ext_score["ami"], best_ext_score["ari"], best_ext_score["silhouette_avg"], best_ext["hyperparams"]))
        
        np.save("{}/best_ext_cluster_predictions_{}.npy".format(args.save_folder,  best_ext["hyperparams"]), best_ext_score["clustering_predictions"])
        plot_GMM(tsne_2d, best_ext_score["clustering_predictions"], n_clusters, args.save_folder, "best_external")
        plot_silhouette_scores(best_ext_score["sample_silhouette_values"], best_ext_score["silhouette_avg"], best_ext_score['clustering_predictions'], n_clusters, args.save_folder, "best_external")
        
        best_ext_scores.append(best_ext_score)
        best_int_score = {}
        
        print("Testing clustering with {}".format(best_int["hyperparams"]))
        test_embedding = best_int['umap'].transform(test_features)
        best_int_score['clustering_predictions'] = best_int['gmm'].predict(test_embedding)

        if cfg.calculate_acc:
            best_int_score["acc"], best_int_score['acc_per_class_total'], best_int_score['acc_per_class_relative'], best_int_score['cluster_class_assigned'] = calculate_clustering_accuracy_expanded_with_overclustering(best_int_score['clustering_predictions'], test_targets, num_classes)
            f = lambda v: np.round(v, 2)
            plot_acc_per_class(test_dataset.classes, f(best_int_score['acc_per_class_total']), 'Best ACC accuracy per class total', 'total', args.save_folder)
            plot_acc_per_class(test_dataset.classes, f(best_int_score['acc_per_class_relative']), 'Best ACC accuracy per class relative', 'relative', args.save_folder)
        else:
            best_int_score['acc'] = -1
        best_int_score["nmi"] = normalized_mutual_info_score(best_int_score['clustering_predictions'], test_targets)
        best_int_score["ami"] = adjusted_mutual_info_score(best_int_score['clustering_predictions'], test_targets)
        best_int_score["ari"] = adjusted_rand_score(best_int_score['clustering_predictions'], test_targets)
        best_int_score["silhouette_avg"] = silhouette_score(test_features, best_int_score['clustering_predictions'])
        best_int_score["sample_silhouette_values"] = silhouette_samples(test_features, best_int_score['clustering_predictions'])

        print("\nBest internal clustering with n_clusters={}".format(n_clusters))
        print("\tACC: {} NMI: {} AMI: {} ARI: {} silhouette: {}\t{}\n".format(best_int_score["acc"], best_int_score["nmi"], best_int_score["ami"], best_int_score["ari"], best_int_score["silhouette_avg"], best_int["hyperparams"]))
        
        np.save("{}/best_int_cluster_predictions_{}.npy".format(args.save_folder,  best_int["hyperparams"]), best_int_score["clustering_predictions"])
        plot_GMM(tsne_2d, best_int_score["clustering_predictions"], n_clusters, args.save_folder, "best_internal")
        plot_silhouette_scores(best_int_score["sample_silhouette_values"], best_int_score["silhouette_avg"], best_int_score['clustering_predictions'], n_clusters, args.save_folder, "best_internal")
    
        best_int_scores.append(best_int_score)
        time_passed = time.time() - start
        print("\tTime passed: ", time.time() - start)

        with open("{}/results.txt".format(args.save_folder), 'a') as file:
            file.write("Best external clustering with n_cluster_{}_{}\n".format(n_clusters, best_ext["hyperparams"]))
            file.write("ACC: {} NMI: {} AMI: {} ARI: {} silhouette: {}\n".format(best_ext_score["acc"], best_ext_score["nmi"], best_ext_score["ami"], best_ext_score["ari"], best_ext_score["silhouette_avg"]))
            file.write("Best internal clustering with n_cluster_{}_{}\n".format(n_clusters, best_int["hyperparams"]))
            file.write("ACC: {} NMI: {} AMI: {} ARI: {} silhouette: {}\n".format(best_int_score["acc"], best_int_score["nmi"], best_int_score["ami"], best_int_score["ari"], best_int_score["silhouette_avg"]))
            file.write("Time passed: {} s\n".format(time_passed))

    # plotting scores
    plot_scores(n_clusters_values, best_ext_scores, original_save_folder, "best_external", cfg)
    plot_scores(n_clusters_values, best_int_scores, original_save_folder, "best_internal", cfg)


def extract_features_targets(model, dataset, cfg):
    features = []
    targets = []

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1, 
                pin_memory=True, drop_last=False)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(loader):
            img = img.cuda()

            feature = model.forward(img)
            if cfg.features_layer == 'layer4':
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
    fig.savefig('{}/t-SNE.svg'.format(save_folder))
    # legendFig.savefig('{}/UMAP_legend.svg'.format(args.save_folder))
    plt.close()


def plot_UMAP(dataset, embedding, targets, save_folder, hyperparams, prefix):
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
    fig.savefig('{}/{}_UMAP_{}.svg'.format(save_folder, prefix, hyperparams))
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


def plot_GMM_with_class_assigned(embedding, predicted_cluster_labels, num_clusters, cluster_class_assigned, classes_names, save_folder):
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
    for i, clusters in enumerate(cluster_class_assigned):
        for c in clusters:
            legend_item = mlines.Line2D([], [], color=np.array(colors_per_cluster[c], dtype=float)/255, marker='o', linestyle='None',
                                markersize=10)
            legend_elements.append(legend_item)
            legend_labels.append('cluster {} -> class {}'.format(c, classes_names[i]))
    ax.legend(legend_elements, legend_labels, bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    fig.savefig('{}/GMM_with_class_assigned.svg'.format(save_folder))
    plt.close()


def plot_acc_per_class(x, y, title, suffix, save_folder):
    plt.figure("Bar Chart", figsize=(12,6))
    plt.xticks(rotation=45, ha='right')
    rect = plt.bar(x, y)
    plt.bar_label(rect, padding=3)

    plt.gca().set(title=title, ylabel="accuracy")
    plt.tight_layout()

    bottom, top = plt.ylim()
    plt.ylim([bottom, top + top*0.2])
    
    plt.savefig("{}/ACC_per_class_{}.svg".format(save_folder, suffix))
    plt.close()


def plot_silhouette_scores(sample_silhouette_values, silhouette_avg, cluster_labels, n_clusters, save_folder, prefix):
    
    fig = plt.figure("Silhouette Scores", figsize=(10,20))
    ax = fig.add_subplot(111)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.savefig("{}/{}silhouette.svg".format(save_folder, prefix + "_" if prefix else ""))
    plt.close()

def plot_scores(n_clusters_values, scores, save_folder, prefix, cfg):
    accs = []
    nmis = []
    amis = []
    aris = []
    silhouettes = []
    
    for i, n in enumerate(n_clusters_values):
        
        if cfg.calculate_acc:
            accs.append(scores[i]['acc'])
        nmis.append(scores[i]['nmi'])
        amis.append(scores[i]['ami'])
        aris.append(scores[i]['ari'])
        silhouettes.append(scores[i]['silhouette_avg'])
        
    fig, ax = plt.subplots(1, 1)
    if cfg.calculate_acc:
        ax.plot(n_clusters_values, accs, label='ACC') 
    ax.plot(n_clusters_values, nmis, label='NMI')
    ax.plot(n_clusters_values, amis, label='AMI')
    ax.plot(n_clusters_values, aris, label='ARI')
    ax.plot(n_clusters_values, silhouettes, label='SILHOUETTE')
   
    ax.set_xlabel('num_cluster')
    ax.set_ylabel('score')
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend(); 
    fig.savefig('{}/{}scores.svg'.format(save_folder, prefix + "_" if prefix else ""))
    plt.close()

if __name__ == '__main__':
    main()
