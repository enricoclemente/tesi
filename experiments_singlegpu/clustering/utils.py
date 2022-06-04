import sys
sys.path.insert(0, './')

import numpy as np
from SPICE.spice.utils.evaluation import calculate_acc


def calculate_clustering_accuracy_expanded(y_pred, y_true, num_classes):

    class_accuracy_total = np.zeros(num_classes)
    class_accuracy_relative = np.zeros(num_classes)
    cluster_class_assigned = np.zeros(num_classes)

    try:
        acc, cluster_labels_assigned, g_truth_labels_assigned = calculate_acc(y_pred, y_true, return_idx=True)
    except AssertionError as msg:
        print("Clustering Accuracy failed: ") 
        print(msg)
        acc = -1
        cluster_class_assigned = np.array([])
        g_truth_labels_assigned = np.array([])
    
    N = len(np.unique(y_pred))

    s = np.unique(y_pred)
    t = np.unique(y_true)

    # get overall accuracy and for every class 
    count = 0
    for i in range(N):
        idx = np.logical_and(y_pred == s[cluster_labels_assigned[i]], y_true == t[g_truth_labels_assigned[i]])
        class_accuracy_total[g_truth_labels_assigned[i]] = np.count_nonzero(idx) / len(y_true.tolist())
        class_accuracy_relative[g_truth_labels_assigned[i]] = np.count_nonzero(idx)/y_true.tolist().count(g_truth_labels_assigned[i])
        cluster_class_assigned[g_truth_labels_assigned[i]] = cluster_labels_assigned[i]
        # print("class: {} has accuracy: {}".format(g_truth_labels_assigned[i], np.count_nonzero(idx)/y_true.tolist().count(g_truth_labels_assigned[i])))
        count += np.count_nonzero(idx) 
    # print(count)
    
    # print(class_accuracy_total)
    # print(class_accuracy_per_class)
    acc = 1.0 * count / len(y_true)
    # print("Accuracy: {}".format(acc))

    return acc, class_accuracy_total, class_accuracy_relative, cluster_class_assigned.astype(int).tolist()