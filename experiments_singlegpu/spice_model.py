import sys
sys.path.insert(0, './')

import torch
import torch.nn as nn
from spice.model.heads import build_head


""" SPICE model for creating prototype pseulabels
    class restructured from sim2sem where instead of having inner choice, the outer script will lead
    the components --> more black box approach --> more mantainable
    - feature_model
    - head

"""
class SPICEModel(nn.Module):
    def __init__(self, feature_model, head):
        super(SPICEModel, self).__init__()

        # encoder for extracting features
        self.feature_module = feature_model

        # clustering head
        self.head = head

        # gradient not computed for encoder
        for param in self.feature_module.parameters():
            param.requires_grad = False
        

    def forward(self, images, targets):
        return 0
    
    # Function compute loss between images and targets
    def loss(self, images, targets):
        if isinstance(images, list):
            features = []
            num_heads = len(images)
            num_each = images[0].shape[0]
            images_all = torch.cat(images, dim=0)
            features_all = self.feature_module(images_all)
            for h in range(num_heads):
                s = h*num_each
                e = s + num_each
                features.append(features_all[s:e, ...])
        else:
            features = self.feature_module(images)

        return self.head.loss(features, targets)
    

    # Function extract features using feature_module
    # returns: tensor [N,C] N=number of images, C=number of features
    def extract_only_features(self, images):
        pool = nn.AdaptiveAvgPool2d(1)

        features = self.feature_module(images)

        if len(features.shape) == 4:
            features = pool(features)
            features = torch.flatten(features, start_dim=1)
        features = nn.functional.normalize(features, dim=1)

        return features


    # Function extracts probabilities that an image belong to a cluster
    # this function is forward with forward_type sem of the original Sim2Sem model
    # returns: tensor [N,K] N=number of images, K=number of cluster
    def sem(self, images):
        if isinstance(images, list):
            features = []
            num_heads = len(images)
            num_each = images[0].shape[0]
            images_all = torch.cat(images, dim=0)
            features_all = self.feature_module(images_all)
            for h in range(num_heads):
                s = h*num_each
                e = s + num_each
                features.append(features_all[s:e, ...])
        else:
            features = self.feature_module(images)

        return self.head.forward(features)
    

    # Function get prototype labels
    # this function is forward with forward_type sim2sem of the original Sim2Sem model
    def sim2sem(self, features, scores, epoch):
        return self.head.select_samples(features, scores, epoch)
    

    # Function get reliable pseudo-labels
    # this function is forward with forward_type local_consistency of the original Sim2Sem model
    def local_consistency(self, features, scores):
        return self.head.local_consistency(features, scores)
        
