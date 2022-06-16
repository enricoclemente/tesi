#!/usr/bin/env python
import sys
import os

sys.path.insert(0, './')

import torch
from PIL import Image
import open_clip
import torch.nn.functional as F
import numpy as np
from experiments_singlegpu.datasets.SocialProfilePictures import SocialProfilePictures
import torchvision.transforms as transforms
import torchvision.models as models

# https://github.com/mlfoundations/open_clip/blob/main/docs/Interacting_with_open_clip.ipynb

clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_e16')
print(clip_model)

resnet_model = models.resnet18(pretrained=True)
resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))

dataset = SocialProfilePictures(root='/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets', version=3, randomize_metadata=True,
                                transform=transforms.Compose([
                                                transforms.Resize([224, 224]),
                                                transforms.ToTensor()]) )

with torch.no_grad():
    for i, (img, target) in enumerate(dataset):

        clip_img = img.unsqueeze(0)
        clip_feature = clip_model.encode_image(clip_img)

        # clip_feature = torch.flatten(clip_feature, 1)
        clip_feature = F.normalize(clip_feature, dim=1)
        # clip_feature /= clip_feature.norm(dim=-1, keepdim=True)
        print("clip extracted feature", clip_feature.shape)


        resnet_img = img.unsqueeze(0)
        resnet_feature = resnet_model(resnet_img)
        resnet_feature = torch.flatten(resnet_feature, 1)
        resnet_feature = F.normalize(resnet_feature, dim=1)

        print("resnet extracted feature", resnet_feature.shape)
        exit()



