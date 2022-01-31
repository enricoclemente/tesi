import torch
import torch.nn as nn
import numpy as np
from ..feature_modules.build_feature_module import build_feature_module



""" It is the Clustering Head for SPICE
    parameters:
    - classifier: dictionary {type=string, num_neurons=list, last_activation=string} how to create the classifier
    - feature_conv: ??? how to create a feature module
    - num_cluster: how many clusters
    - center_ratio: ??? ratio for selecting samples to use as clustering centers
    - iter_start: ???
    - iter_up: ???
    - iter_down: ???
    - iter_end: ???
    - loss_weight: dictionary{loss_cls=int, loss_ent=int} ??? to slow down learning
    - fea_fc: ??? if the features passed are from fc layers 
    - T: temperature
    - sim_ratio: ??? never used
    - sim_center_ratio: ??? never used
    - epoch_merge: ??? never used
    - entropy: ??? 

    attributes: as parameters
"""
class SemHead(nn.Module):
    def __init__(self, classifier, feature_conv=None, num_cluster=10, center_ratio=0.5,
                 iter_start=0, iter_up=-1, iter_down=-1, iter_end=0, ratio_start=0.5, ratio_end=0.95, 
                 loss_weight=None, fea_fc=False, T=1, sim_ratio=1, sim_center_ratio=0.9, epoch_merge=5, 
                 entropy=False):

        super(SemHead, self).__init__()

        if loss_weight is None:
            loss_weight = dict(loss_cls=1, loss_ent=0)
        self.loss_weight = loss_weight

        self.classifier = build_feature_module(classifier)

        self.feature_conv = None
        if feature_conv:
            self.feature_conv = build_feature_module(feature_conv)

        self.num_cluster = num_cluster
        self.loss_fn_cls = nn.CrossEntropyLoss()
        self.iter_start = iter_start
        self.iter_end = iter_end
        self.ratio_start = ratio_start
        self.ratio_end = ratio_end
        self.center_ratio = center_ratio
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.fea_fc = fea_fc
        self.T = T
        self.sim_ratio = sim_ratio
        self.iter_up = iter_up
        self.iter_down = iter_down
        self.sim_center_ratio = sim_center_ratio
        self.epoch_merge = epoch_merge

        self.entropy = entropy
        self.EPS = 1e-5     # related to entropy

    """
    def compute_ratio_selection_old(self, i):
        if self.ratio_end == self.ratio_start:
            return self.ratio_start
        elif self.iter_start < i <= self.iter_end:
            r = (self.ratio_end - self.ratio_start) / (self.iter_end - self.iter_start) * (i - self.iter_start) + self.ratio_start
            return r
        else:
            return self.ratio_start
    """

    def compute_ratio_selection(self, i):
        if self.ratio_end == self.ratio_start:
            return self.ratio_start
        elif self.iter_up != -1 and self.iter_down != -1:
            if i < self.iter_start:
                return self.ratio_start
            elif self.iter_start <= i < self.iter_up:
                r = (self.ratio_end - self.ratio_start) / (self.iter_up - self.iter_start) * (i - self.iter_start) + self.ratio_start
                return r
            elif self.iter_up <= i < self.iter_down:
                return self.ratio_end
            elif self.iter_down <= i < self.iter_end:
                r = self.ratio_end - (self.ratio_end - self.ratio_start) / (self.iter_end - self.iter_down) * (i - self.iter_down)
                return r
            else:
                return self.ratio_start
        else:
            if self.iter_start < i <= self.iter_end:
                r = (self.ratio_end - self.ratio_start) / (self.iter_end - self.iter_start) * (i - self.iter_start) + self.ratio_start
                return r
            else:
                return self.ratio_start
    """
    def select_samples_cpu(self, feas_sim, scores, i):

        # sort scores and get indices of the sorted scores
        _, idx_max = torch.sort(scores, dim=0, descending=True)
        idx_max = idx_max.cpu()
        num_per_cluster = idx_max.shape[0] // self.num_cluster
        ratio_select = self.compute_ratio_selection(i)

        k = int(self.center_ratio * num_per_cluster * ratio_select)
        print(k)
        idx_max = idx_max[0:k, :]

        centers = []
        for c in range(self.num_cluster):
            centers.append(feas_sim[idx_max[:, c], :].mean(axis=0))

        select_idx_all = []
        select_labels_all = []
        num_per_cluster = feas_sim.shape[0] // self.num_cluster
        ratio_select = self.compute_ratio_selection(i)
        num_select_c = int(num_per_cluster * ratio_select)
        for c in range(self.num_cluster):
            center_c = centers[c]
            dis = np.dot(feas_sim, center_c.T).squeeze()
            idx_s = np.argsort(dis)[::-1]
            idx_select = idx_s[0:num_select_c]

            select_idx_all = select_idx_all + list(idx_select)
            select_labels_all = select_labels_all + [c] * len(idx_select)

        select_idx_all = np.array(select_idx_all)
        select_labels_all = np.array(select_labels_all)

        return select_idx_all, select_labels_all
    """
    # select the samples that get the best scores for every cluster --> prototype labels
    # return tuple (images selected as prototype, in which cluster prototypes have been classified)
    def select_samples(self, features, scores, i):

        # sort scores by column --> obtain: 
        # - sorted scores per cluster (_)
        # - for every cluster get the information of the image that get get the best scores sorted (indices)
        _, indices = torch.sort(scores, dim=0, descending=True)
        indices = indices.cpu()
        
        # get number of scores per cluster
        num_per_cluster = indices.shape[0] // self.num_cluster

        # TODO cos'è??? per cifar10 è sempre 1 dato che ratio_end == ratio_start nel file di conf
        ratio_select = self.compute_ratio_selection(i)
        # print(ratio_select)
        k = int(self.center_ratio * num_per_cluster * ratio_select)
        # print(k, len(idx_max))
        
        # get first k tensor of indices 
        indices_topk = indices[0:k, :]

        centers = []
        for c in range(self.num_cluster):
            # select for every cluster the sample(s) with the best scores 
            # and take the corresping features
            # make mean by columns extracting a one-dimensional tensor [C]
            # unsqueeze it obtaining a tensor [1,C] C, is dimension of features
            centers.append(features[indices_topk[:, c], :].mean(axis=0).unsqueeze(dim=0))

        # retrieve a unique tensor [K,C]
        centers = torch.cat(centers, dim=0)


        num_samples_to_select = int(num_per_cluster * ratio_select)

        # calculate cosine similarity between cluster centers and features to obtain the nearest 
        # samples for every center so they will have the same cluster label
        # in other words, for every cluster the score for every image
        # tensor [K, D]
        cosine_similarity = torch.einsum('cd,nd->cn', [centers, features])

        # take the first num_samples_to_select indices of the best similarities and flatten them into a tensor [K*num_samples_to_select]
        # in other words for every cluster take the first num_samples_to_select indices that refers to those samples
        # that achieved the best similarities with each clustering center
        similarity_best_indices = torch.argsort(cosine_similarity, dim=1, descending=True)[:, 0:num_samples_to_select].flatten()

        # create tensor of values [0, 1, ... K-1]
        # unsqueeze it into tensor [k,1] (column with unique values)
        # enlarge every row tensor replicating values num_samples_to_select times --> tensor [K, num_samples_to_select]
        # transorm again tensor into a one dimension tensor [K*num_samples_to_select] --> [0,0,0, .. 1,1,1, .. 2,2,2, .....]
        # this tensor is useful because in the training script we don't have information of how many images are per cluster
        clusters_labels_to_select = torch.arange(0, self.num_cluster).unsqueeze(dim=1).repeat(1, num_samples_to_select).flatten()

        return similarity_best_indices, clusters_labels_to_select

    """
    def select_samples_v2(self, feas_sim, scores, i):

        _, idx_max = torch.sort(scores, dim=0, descending=True)
        idx_max = idx_max.cpu()
        num_per_cluster = idx_max.shape[0] // self.num_cluster
        ratio_select = self.compute_ratio_selection(i)
        # print(ratio_select)
        k = int(self.center_ratio * num_per_cluster * ratio_select)
        # print(k, len(idx_max))

        idx_center_exist = torch.zeros_like(idx_max[:, 0], dtype=torch.bool)

        centers = []
        for c in range(self.num_cluster):
            idx_c = idx_max[:, c]
            if c == 0:
                idx_c_select = idx_c[0:k]
            else:
                idx_c_available = ~idx_center_exist[idx_c]
                idx_c_select = idx_c[idx_c_available][0:k]

            idx_center_exist[idx_c_select] = True

            centers.append(feas_sim[idx_c_select, :].mean(axis=0).unsqueeze(dim=0))

        centers = torch.cat(centers, dim=0)

        num_select_c = int(num_per_cluster * ratio_select)

        dis = torch.einsum('cd,nd->cn', [centers, feas_sim])
        # idx_select = torch.argsort(dis, dim=1, descending=True)[:, 0:num_select_c].flatten()
        idx_sort = torch.argsort(dis, dim=1, descending=True)
        idx_label_exist = torch.zeros_like(idx_sort[0, :], dtype=torch.bool)
        labels_select_all = []
        idx_select_all = []
        for c in range(self.num_cluster):
            idx_c = idx_sort[c, :]
            if c == 0:
                idx_c_select = idx_sort[0, 0:num_select_c]
            else:
                idx_c_available = ~idx_label_exist[idx_c]
                idx_c_select = idx_c[idx_c_available][0:num_select_c]

            idx_label_exist[idx_c_select] = True
            idx_select_all.append(idx_c_select)
            labels_select_all.append(torch.zeros_like(idx_c_select)+c)

        idx_select_all = torch.cat(idx_select_all)
        labels_select_all = torch.cat(labels_select_all)
        print(len(set(idx_select_all.cpu().numpy())))

        return idx_select_all, labels_select_all
    """

    # get cluster scores
    def forward(self, features):

        if self.feature_conv is not None:
            features = self.feature_conv(features)

        if not self.fea_fc:
            # if features have rank > 2, flatten them
            features = self.avg_pooling(features) 
            features = features.flatten(start_dim=1)

        cluster_score = self.classifier(features)

        cluster_score = cluster_score / self.T

        return cluster_score

    # get loss 
    def loss(self, x, target):
        # calculate cluster score
        clustering_score = self.forward(x)

        # calculate cross entropy loss between scores and prototype labels
        loss = self.loss_fn_cls(clustering_score, target) * self.loss_weight["loss_cls"]

        # for spice training is never executed
        if self.entropy:
            prob_mean = clustering_score.mean(dim=0)
            prob_mean[(prob_mean < self.EPS).data] = self.EPS
            loss_ent = (prob_mean * torch.log(prob_mean)).sum()
            loss = loss + loss_ent * self.loss_weight["loss_ent"]

        return loss
