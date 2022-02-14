import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spice.model.heads.sem_head import SemHead


""" Wrapper for several Clustering Heads
    parameters:
    - multi_head: list(dictionary{...}, ....dictionary{...}) 
                    for its structure see SemHead class which will use the elements of the list
    - ratio_confident: ???
    - num_neighbor: ???
    - score_th: ???

    attributes:
    - num_heads: number of clustering heads 
    - num_cluster: number of clusters
    - center_ratio:  ??? ratio for selecting samples to use as clustering centers
    - num_neighbor: ???
    - ratio_confident: ???
    - score_th: ???
    - head_[h]: clustering head (SemHead)
"""
class SemHeadMulti(nn.Module):
    def __init__(self, multi_heads, ratio_confident=0.90, num_neighbor=100, score_th=0.99, **kwargs):

        super(SemHeadMulti, self).__init__()

        self.num_heads = len(multi_heads) 
        self.num_cluster = multi_heads[0].num_cluster   # use number of classes if you know
        self.center_ratio = multi_heads[0].center_ratio # ? 
        self.num_neighbor = num_neighbor
        self.ratio_confident = ratio_confident
        self.score_th = score_th
        for h in range(self.num_heads):
            head_h = SemHead(**multi_heads[h])
            self.__setattr__("head_{}".format(h), head_h)

    # Function get reliable samples
    # return the reliable images and relative labels (classified clusters)
    # 
    def local_consistency(self, features, scores):
        # take the best scores = cluster label predictions
        labels_pred = scores.argmax(dim=1).cpu()
        # compute similarity matrix between features and features
        # tensor [N,N]
        sim_mtx = torch.einsum('nd,cd->nc', [features.cpu(), features.cpu()])
        # the top k will retrieve in this case as first element always the diagonal indices
        scores_k, idx_k = sim_mtx.topk(k=self.num_neighbor, dim=1)
        
        # take for every sample the neighbors and their prediction,
        # the first column will be the image itself because 
        labels_samples = torch.zeros_like(idx_k)
        for s in range(self.num_neighbor):
            labels_samples[:, s] = labels_pred[idx_k[:, s]]

        # get neighbors that have been classified in the same cluster
        true_mtx = labels_samples[:, 0:1] == labels_samples
        # how many neighbors have been classified in the same cluster (including the image itself)
        num_true = true_mtx.sum(dim=1)
        
        # which samples have the number of neighbors with the same cluster that are over a certain threshold
        idx_true = num_true >= self.num_neighbor * self.ratio_confident
        # print(num_true.min())

        # confident samples: which samples get the clustering score above score_th value
        idx_conf = scores.max(dim=1)[0].cpu() > self.score_th
        # get if confident samples have also a certain number of neighbor classified in the same cluster
        idx_true = idx_true * idx_conf
        # select the samples which are both confident and with also neighbor in the same cluster
        idx_select = torch.where(idx_true > 0)[0]
        # select the clustering predictions of hte previous selected samples
        labels_select = labels_pred[idx_select]

        num_per_cluster = []
        idx_per_cluster = []
        label_per_cluster = []
        for c in range(self.num_cluster):
            # for the i-th cluster which selected images belong to that cluster
            idx_c = torch.where(labels_select == c)[0]
            # put for the i-th cluster the relative images classified in
            idx_per_cluster.append(idx_select[idx_c])
            num_per_cluster.append(len(idx_c))
            # put for the i-th cluster the relative labels (cluster index) of the selected images
            label_per_cluster.append(labels_select[idx_c])

        idx_per_cluster_select = []
        label_per_cluster_select = []
        # which cluster has the smallest number of samples
        min_cluster = np.array(num_per_cluster).min()
        # for every cluster select min_cluster images and shuffle the order of them
        for c in range(self.num_cluster):
            idx_shuffle = np.arange(0, num_per_cluster[c])
            np.random.shuffle(idx_shuffle)
            idx_per_cluster_select.append(idx_per_cluster[c][idx_shuffle[0:min_cluster]])
            label_per_cluster_select.append(label_per_cluster[c][idx_shuffle[0:min_cluster]])

        idx_select = torch.cat(idx_per_cluster_select)
        labels_select = torch.cat(label_per_cluster_select)

        # return the reliable images and relative labels (classified clusters)
        return idx_select, labels_select

    # get cluster centers
    def compute_cluster_proto(self, feas_sim, scores):

        _, idx_max = torch.sort(scores, dim=0, descending=True)
        idx_max = idx_max.cpu()
        num_per_cluster = idx_max.shape[0] // self.num_cluster
        k = int(self.center_ratio * num_per_cluster)
        idx_max = idx_max[0:k, :]
        centers = []
        for c in range(self.num_cluster):
            centers.append(feas_sim[idx_max[:, c], :].mean(axis=0).unsqueeze(dim=0))

        centers = torch.cat(centers, dim=0)
        return centers


    # get prototype labels for every cluster head
    def select_samples(self, features, scores, i):
        assert len(scores) == self.num_heads
        
        prototypes_indices = []
        prototype_cluster_labels = []
        for h in range(self.num_heads):
            prototype_samples_indices, gt_cluster_labels = self.__getattr__("head_{}".format(h)).select_samples(features, scores[h], i)
            prototypes_indices.append(prototype_samples_indices)
            prototype_cluster_labels.append(gt_cluster_labels)

        return prototypes_indices, prototype_cluster_labels


    # Function get for every head cluster score
    # returns: list of tensors [N,K] 
    #           N=number of images k=number of clusters, 
    #           len of list == number of clustering heads
    def forward(self, features):
        cluster_scores = []
        if isinstance(features, list):
            assert len(features) == self.num_heads

        for h in range(self.num_heads):
            if isinstance(features, list):
                cluster_score_h = self.__getattr__("head_{}".format(h)).forward(features[h])
            else:
                # tensor [N, K], K is number of clusters
                cluster_score_h = self.__getattr__("head_{}".format(h)).forward(features)

            cluster_scores.append(cluster_score_h)

        return cluster_scores


    # Function loss from every cluster head
    # returns: list of losses
    def loss(self, x, target):
        assert len(x) == self.num_heads
        assert len(target) == self.num_heads

        loss = {}
        for h in range(self.num_heads):
            loss_h = self.__getattr__("head_{}".format(h)).loss(x[h], target[h])
            loss['head_{}'.format(h)] = loss_h

        return loss
