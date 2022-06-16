#!/usr/bin/env python
import sys
import os
from typing import OrderedDict
sys.path.insert(0, './')

import numpy as np
import matplotlib.pyplot as plt



def aggregate_clustering_results_and_plot(root, folders, save_folder, prefix):

    nmis = np.zeros(73)
    amis = np.zeros(73)
    aris = np.zeros(73)
    silhouette_avgs = np.zeros(73)

    for folder in folders:
        confs = os.listdir(os.path.join(root, folder))
        for conf in confs:
            print(conf)
            
            if os.path.isdir(os.path.join(root, folder, conf)):
                results_file = os.path.join(root, folder, conf, 'results.txt')
                with open(results_file, 'r') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        if i == 3:
                            print(line)
                            nmi = float(line.split(':')[2].split(" ")[1])
                            ami = float(line.split(':')[3].split(" ")[1])
                            ari = float(line.split(':')[4].split(" ")[1])
                            silhouette_avg = float(line.split(':')[5].split(" ")[1])
                            print(nmi, ami, ari, silhouette_avg)
                            nmis[int(conf.split("_")[2])] = nmi
                            amis[int(conf.split("_")[2])] = ami
                            aris[int(conf.split("_")[2])] = ari
                            silhouette_avgs[int(conf.split("_")[2])] = silhouette_avg
    
    real_nmis = []
    real_amis = []
    real_aris = []
    real_silhouettes = []
    n_clusters_values = []

    for i in range(73):
        if nmis[i] > 0.0:
            real_nmis.append(nmis[i])
            real_amis.append(amis[i])
            real_aris.append(aris[i])
            real_silhouettes.append(silhouette_avgs[i])
            n_clusters_values.append(i) 

    fig, ax = plt.subplots(1, 1, figsize=(15,7))

    ax.plot(n_clusters_values, real_nmis, label='NMI')
    ax.plot(n_clusters_values, real_amis, label='AMI')
    ax.plot(n_clusters_values, real_aris, label='ARI')
    ax.plot(n_clusters_values, real_silhouettes, label='SILHOUETTE')
   
    ax.set_xlabel('num_cluster')
    ax.set_ylabel('score')
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(2, 74, 2).tolist())
    ax.grid(True)
    ax.legend(); 
    fig.savefig('{}/{}scores.svg'.format(save_folder, prefix))
    plt.close()

def main():
    aggregate_clustering_results_and_plot('/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/version_03/clustering/umap+gmm/moco_both_encoders_pretrained_exp2',
                                        ['exp1', 'exp2'], '/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/version_03/clustering/analysis', "moco_both_encoders_pretrained_exp2_")

    aggregate_clustering_results_and_plot('/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/version_03/clustering/umap+gmm/resnet18_pretrained',
                                    ['exp2', 'exp4'], '/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/version_03/clustering/analysis', "resnet18_pretrained_")

if __name__ == '__main__':
    main()