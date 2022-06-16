#!/usr/bin/env python
import sys
import os
from click import version_option
sys.path.insert(0, './')

import time
from math import sqrt
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import scipy.io
from scipy.optimize import linear_sum_assignment
import numpy as np
from itertools import combinations_with_replacement
import multiprocessing as mp
from joblib import Parallel, delayed
import tqdm

    

def clustering_accuracy_spice_overcluster(y_pred, y_true):
    print("Cluster Accuracy SPICE with overcluster")
    print("Cluster prediction: {}".format(y_pred))
    print("Ground truth: {}".format(y_true))

    cluster_labels = np.unique(y_pred) #s
    class_labels = np.unique(y_true) #t

    if len(cluster_labels) > len(class_labels):
        best_combination = {'acc': 0.0}

        overcluster = len(cluster_labels) - len(class_labels)
        class_extra_combinations = combinations_with_replacement(class_labels, overcluster)

        for comb in class_extra_combinations:
            print("comb", comb)
            class_combination = np.concatenate([class_labels, comb])
            print("Class combination: ", class_combination)

            # calculatin cost matrix for i-th combination
            N = len(cluster_labels)
            C = np.zeros((N, N), dtype=np.int32)

            for i in range(N):
                for j in range(N):
                    idx = np.logical_and(y_pred == cluster_labels[i], y_true == class_combination[j])
                    # print("For cluster {} and class {}".format(i, j))
                    # print(y_pred == cluster_labels[i])
                    # print(y_true == class_combination[j])
                    # print(idx)
                    C[i][j] = np.count_nonzero(idx)
            Cmax = np.amax(C)
            C = Cmax - C
            print("Cost Matrix: \n{}".format(C))

            row, col = linear_sum_assignment(C)

            col_fake_assignement = col.copy()

            for i, a in enumerate(col_fake_assignement):
                col[i] = class_combination[a]
            print("Linear sum assignment results")
            print("Cluster label: {}".format(row))
            print("Fake assigned ground truth label: {}".format(col_fake_assignement))
            print("Real assigned ground truth label: {}".format(col))

            count = 0
            for i in range(N):
                idx = np.logical_and(y_pred == cluster_labels[row[i]], y_true == class_combination[col[i]])
                print("cluster: {} assigned to class: {} gives accuracy: {}".format(row[i], col[i], np.count_nonzero(idx)/y_true.tolist().count(col[i])))
                count += np.count_nonzero(idx) 
    
            acc = 1.0 * count / len(y_true)
            print("Accuracy: {}".format(acc))

            if acc > best_combination["acc"]:
                best_combination["acc"] = acc
                best_combination['cluster_labels_assigned'] = row
                best_combination['g_truth_labels_assigned'] = col
        
        print("Best combination is: ", best_combination)
        return best_combination["acc"], best_combination["cluster_labels_assigned"], best_combination["g_truth_labels_assigned"]

    elif len(cluster_labels) == len(class_labels):
        N = len(cluster_labels)
        C = np.zeros((N, N), dtype=np.int32)

        for i in range(N):
            for j in range(N):
                idx = np.logical_and(y_pred == cluster_labels[i], y_true == class_labels[j])
                # print("For cluster {} and class {}".format(i, j))
                # print(y_pred == cluster_labels[i])
                # print(y_true == class_labels[j])
                # print(idx)
                C[i][j] = np.count_nonzero(idx)
        Cmax = np.amax(C)
        C = Cmax - C
        print("Cost Matrix: \n{}".format(C))

        """
            Return an array of row indices and one of corresponding column indices giving the optimal assignment. 
            The cost of the assignment can be computed as cost_matrix[row_ind, col_ind].sum(). 
        """
        row, col = linear_sum_assignment(C)
        
        print("Linear sum assignment results")
        print("Cluster label: {}".format(row))
        print("Relative best ground truth label: {}".format(col))

        # print(C[row,col].sum())
        
        # calcolo il cluster migliore
        count = 0
        for i in range(N):
            idx = np.logical_and(y_pred == cluster_labels[row[i]], y_true == class_labels[col[i]])
            print("class: {} has accuracy: {}".format(col[i], np.count_nonzero(idx)/y_true.tolist().count(col[i%len(np.unique(y_true))])))
            count += np.count_nonzero(idx) 
        # print(count)
        
        acc = 1.0 * count / len(y_true)
        print("Accuracy: {}".format(acc))

        return acc, row, col
    else:
        return -1, [], []

def compute_linear_assignment(comb, class_labels, cluster_labels):

    class_combination = np.concatenate([class_labels, comb])
    print("Class combination: ", class_combination)

    # calculatin cost matrix for i-th combination
    N = len(cluster_labels)
    C = np.zeros((N, N), dtype=np.int32)

    for i in range(N):
        for j in range(N):
            idx = np.logical_and(y_pred == cluster_labels[i], y_true == class_combination[j])
            # print("For cluster {} and class {}".format(i, j))
            # print(y_pred == cluster_labels[i])
            # print(y_true == class_combination[j])
            # print(idx)
            C[i][j] = np.count_nonzero(idx)
    Cmax = np.amax(C)
    C = Cmax - C
    # print("Cost Matrix: \n{}".format(C))

    row, col = linear_sum_assignment(C)

    col_fake_assignement = col.copy()

    for i, a in enumerate(col_fake_assignement):
        col[i] = class_combination[a]
    print("Linear sum assignment results")
    print("Cluster label: {}".format(row))
    print("Fake assigned ground truth label: {}".format(col_fake_assignement))
    print("Real assigned ground truth label: {}".format(col))

    count = 0
    for i in range(N):
        idx = np.logical_and(y_pred == cluster_labels[row[i]], y_true == class_combination[col[i]])
        print("cluster: {} assigned to class: {} gives accuracy: {}".format(row[i], col[i], np.count_nonzero(idx)/y_true.tolist().count(col[i])))
        count += np.count_nonzero(idx) 

    acc = 1.0 * count / len(y_true)
    print("Accuracy: {}".format(acc))
    return acc, row, col


def clustering_accuracy_spice_overcluster_parallel(y_pred, y_true):
    print("Cluster Accuracy SPICE with overcluster")
    print("Cluster prediction: {}".format(y_pred))
    print("Ground truth: {}".format(y_true))

    cluster_labels = np.unique(y_pred) #s
    class_labels = np.unique(y_true) #t

    print("Class labels", class_labels)
   
    if len(cluster_labels) > len(class_labels):
        best_combination = {'acc': 0.0}

        overcluster = len(cluster_labels) - len(class_labels)
        class_extra_combinations = combinations_with_replacement(class_labels, overcluster)
  
        results = Parallel(n_jobs=mp.cpu_count())(delayed(compute_linear_assignment)(i, class_labels, cluster_labels) 
                                                        for i in class_extra_combinations)
        
        for res in results:
            if res[0] > best_combination["acc"]:
                best_combination["acc"] = res[0]
                best_combination['cluster_labels_assigned'] = res[1]
                best_combination['g_truth_labels_assigned'] = res[2]

        
        print("Best combination is: ", best_combination)
        return best_combination["acc"], best_combination["cluster_labels_assigned"], best_combination["g_truth_labels_assigned"]

    elif len(cluster_labels) == len(class_labels):
        N = len(cluster_labels)
        C = np.zeros((N, N), dtype=np.int32)

        for i in range(N):
            for j in range(N):
                idx = np.logical_and(y_pred == cluster_labels[i], y_true == class_labels[j])
                print("For cluster {} and class {}".format(i, j))
                print(y_pred == cluster_labels[i])
                print(y_true == class_labels[j])
                print(idx)
                C[i][j] = np.count_nonzero(idx)
        Cmax = np.amax(C)
        C = Cmax - C
        print("Cost Matrix: \n{}".format(C))

        """
            Return an array of row indices and one of corresponding column indices giving the optimal assignment. 
            The cost of the assignment can be computed as cost_matrix[row_ind, col_ind].sum(). 
        """
        row, col = linear_sum_assignment(C)
        
        print("Linear sum assignment results")
        print("Cluster label: {}".format(row))
        print("Relative best ground truth label: {}".format(col))

        # print(C[row,col].sum())
        
        # calcolo il cluster migliore
        count = 0
        for i in range(N):
            idx = np.logical_and(y_pred == cluster_labels[row[i]], y_true == class_labels[col[i]])
            print("class: {} has accuracy: {}".format(col[i], np.count_nonzero(idx)/y_true.tolist().count(col[i%len(np.unique(y_true))])))
            count += np.count_nonzero(idx) 
        # print(count)
        
        acc = 1.0 * count / len(y_true)
        print("Accuracy: {}".format(acc))

        return acc, row, col
    else:
        return -1, [], []


if __name__ == '__main__':

    y_pred = np.array([1,0,3,4,2,2,2,2,2,3,3])
    y_true = np.array([0,0,0,0,0,0,0,1,1,2,2])
    
    clustering_accuracy_spice_overcluster(y_pred, y_true)
    
