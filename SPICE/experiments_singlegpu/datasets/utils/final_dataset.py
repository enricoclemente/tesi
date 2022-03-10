#!/usr/bin/env python
import sys
import random
import os
sys.path.insert(0, './')

import torch
from experiments_singlegpu.datasets.SocialProfilePictures import create_images_csv
from experiments_singlegpu.datasets.SocialProfilePictures import SocialProfilePictures

import torchvision.transforms as transforms
from experiments_singlegpu.datasets.utils.custom_transforms import PadToSquare

import matplotlib.pyplot as plt


# create_images_csv()

dataset = SocialProfilePictures(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets', 
                transform=transforms.Compose([ PadToSquare(), transforms.Resize([225, 225]), transforms.ToTensor()]))

loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=1, 
            pin_memory=True, drop_last=False)

print("Dataset is big: {} images".format(len(dataset)))
size = 0
# random list to print some images for debug
randomlist = []
for i in range(0,20):
    n = random.randint(1,len(loader))
    randomlist.append(n)
randomlist[0] = 0

classes = dataset.classes
metadata = dataset.metadata

size = 0
for i, (img, target) in enumerate(loader):
    img, target = img.cuda(), target
    if i in randomlist:
        # print(metadata[size]['img_folder'])
        # exit()
        print("Image #{} is a {} from split: {} file path: {}/{}".format(size, metadata[size]['levels']['target_level'],  metadata[size]['split'], metadata[size]['img_folder'], metadata[size]['img_name'] ))
        fig, ax = plt.subplots()
        ax.imshow(img[0].cpu().numpy().transpose([1, 2, 0]))
        labels_text = target[0]
        plt.gcf().text(0.1+(0.4), 0.02, labels_text, fontsize=8)
        plt.savefig(os.path.join('/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/statistics/SocialProfilePictures/version_01/images_examples', "image_{}.jpg".format(size-1)))
        plt.close()
    size += img.size()[0]

    print("[{}]/[{}] batch iteration".format(i, len(loader)))

