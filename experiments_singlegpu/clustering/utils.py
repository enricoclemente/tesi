import sys
sys.path.insert(0, './')

import numpy as np
from SPICE.spice.utils.evaluation import calculate_acc

from scipy.optimize import linear_sum_assignment
from itertools import combinations_with_replacement
import multiprocessing as mp
from joblib import Parallel, delayed


def calculate_clustering_accuracy_expanded(y_pred, y_true, num_classes):

    class_accuracy_total = np.zeros(num_classes)
    class_accuracy_relative = np.zeros(num_classes)
    cluster_class_assigned = np.zeros(num_classes)

    try:
        acc, cluster_labels_assigned, g_truth_labels_assigned = calculate_acc(y_pred, y_true, return_idx=True)
    except AssertionError as msg:
        print("Clustering Accuracy failed: ") 
        print(msg)
        return -1, np.array([]), np.array([]), []
    
    N = len(np.unique(y_pred))

    s = np.unique(y_pred)
    t = np.unique(y_true)

    # get overall accuracy and for every class 
    for i in range(N):
        idx = np.logical_and(y_pred == s[cluster_labels_assigned[i]], y_true == t[g_truth_labels_assigned[i]])
        class_accuracy_total[g_truth_labels_assigned[i]] = np.count_nonzero(idx) / len(y_true.tolist())
        class_accuracy_relative[g_truth_labels_assigned[i]] = np.count_nonzero(idx)/y_true.tolist().count(g_truth_labels_assigned[i])
        cluster_class_assigned[g_truth_labels_assigned[i]] = cluster_labels_assigned[i]
        # print("class: {} has accuracy: {}".format(g_truth_labels_assigned[i], np.count_nonzero(idx)/y_true.tolist().count(g_truth_labels_assigned[i])))

    return acc, class_accuracy_total, class_accuracy_relative, cluster_class_assigned.astype(int).tolist()


def calculate_clustering_accuracy_expanded_with_overclustering(y_pred, y_true, num_classes):

    class_accuracy_total = np.zeros(num_classes)
    class_accuracy_relative = np.zeros(num_classes)
    cluster_class_assigned = []
    for c in range(num_classes):
        cluster_class_assigned.append([])

    try:
        acc, cluster_labels_assigned, g_truth_labels_assigned = calculate_acc_overclustering(y_pred, y_true, return_idx=True, parallelize=True)
    except (AssertionError, NotImplementedError) as msg:
        print("Clustering Accuracy failed: ") 
        print(msg)
        return -1, np.array([]), np.array([]), []
    
    N = len(np.unique(y_pred))

    s = np.unique(y_pred)
    t = np.unique(y_true)

    # get overall accuracy and for every class 
    for i in range(N):
        idx = np.logical_and(y_pred == s[cluster_labels_assigned[i]], y_true == t[g_truth_labels_assigned[i]])
        class_accuracy_total[g_truth_labels_assigned[i]] += np.count_nonzero(idx) / len(y_true.tolist())
        class_accuracy_relative[g_truth_labels_assigned[i]] += np.count_nonzero(idx)/y_true.tolist().count(g_truth_labels_assigned[i]) 
        cluster_class_assigned[g_truth_labels_assigned[i]].append(cluster_labels_assigned[i])

    return acc, class_accuracy_total, class_accuracy_relative, cluster_class_assigned


def calculate_linear_assignment(comb, y_pred, y_true, cluster_labels, class_labels):
    class_combination = np.concatenate([class_labels, comb])

    # calculatin cost matrix for i-th combination
    N = len(cluster_labels)
    C = np.zeros((N, N), dtype=np.int32)

    for i in range(N):
        for j in range(N):
            idx = np.logical_and(y_pred == cluster_labels[i], y_true == class_combination[j])
            C[i][j] = np.count_nonzero(idx)
    Cmax = np.amax(C)
    C = Cmax - C
    
    row, col = linear_sum_assignment(C)
    col_fake_assignement = col.copy()

    for i, a in enumerate(col_fake_assignement):
        col[i] = class_combination[a]

    count = 0
    for i in range(N):
        idx = np.logical_and(y_pred == cluster_labels[row[i]], y_true == class_combination[col[i]])
        count += np.count_nonzero(idx) 

    acc = 1.0 * count / len(y_true)

    return acc, row, col

def calculate_acc_overclustering(y_pred, y_true, return_idx=False, parallelize=False):
    cluster_labels = np.unique(y_pred) #s
    class_labels = np.unique(y_true) #t

    if len(cluster_labels) > len(class_labels):
        best_combination = {'acc': 0.0}

        overcluster = len(cluster_labels) - len(class_labels)
        class_extra_combinations = combinations_with_replacement(class_labels, overcluster)
        results = []
        if parallelize:
            results = Parallel(n_jobs=mp.cpu_count())(delayed(calculate_linear_assignment)(i, y_pred, y_true, cluster_labels, class_labels) 
                                                            for i in class_extra_combinations)
        else:
            for i in class_extra_combinations:
                res = calculate_linear_assignment(i, y_pred, y_true, cluster_labels, class_labels)
                results.append(res)

        for res in results:
            if res[0] > best_combination["acc"]:
                best_combination["acc"] = res[0]
                best_combination['cluster_labels_assigned'] = res[1]
                best_combination['g_truth_labels_assigned'] = res[2]
        
        if return_idx:
            return best_combination["acc"], best_combination["cluster_labels_assigned"], best_combination["g_truth_labels_assigned"]
        else:
            return best_combination["acc"]
    elif len(cluster_labels) == len(class_labels):
        N = len(cluster_labels)
        C = np.zeros((N, N), dtype=np.int32)

        for i in range(N):
            for j in range(N):
                idx = np.logical_and(y_pred == cluster_labels[i], y_true == class_labels[j])
                C[i][j] = np.count_nonzero(idx)
        Cmax = np.amax(C)
        C = Cmax - C

        """
            Return an array of row indices and one of corresponding column indices giving the optimal assignment. 
            The cost of the assignment can be computed as cost_matrix[row_ind, col_ind].sum(). 
        """
        row, col = linear_sum_assignment(C)

        count = 0
        for i in range(N):
            idx = np.logical_and(y_pred == cluster_labels[row[i]], y_true == class_labels[col[i]])
            count += np.count_nonzero(idx) 
        # print(count)
        
        acc = 1.0 * count / len(y_true)

        if return_idx:
            return acc, row, col
        else:
            return acc
    else:
        raise NotImplementedError("Number of cluster is lower than number of classes!")


"""

    Overclustering previsions time:
    n = 24
    - k = 26 - 24 = 2 --> combs = 300       tempo = 2.8 s
    - k = 28 - 24 = 4 --> combs = 17750     tempo = 32 s    
    - k = 30 - 24 = 6 --> combs = 475020    tempo = 900 s 
    - k = 32 - 24 = 8 --> combs = 7888725   tempo stimato = 900 * (7888725 / 475020) = 15000 s (4 h)

"""