#!/usr/bin/env python
import sys
import argparse
import random
import os
import shutil
sys.path.insert(0, './')

import torch
from experiments_singlegpu.datasets.SUN397_custom import SUN397
from experiments_singlegpu.datasets.SelfieImageDetectionDataset_custom import SIDD
from experiments_singlegpu.datasets.OxfordIIITPet_custom import OxfordIIITPet
from experiments_singlegpu.datasets.IIITCFW_custom import CFW
from experiments_singlegpu.datasets.ArtImages_custom import ArtImages
from experiments_singlegpu.datasets.iCartoonFace_custom import iCartoonFace
from experiments_singlegpu.datasets.EMOTIC_custom import EMOTIC

import torchvision.transforms as transforms
from experiments_singlegpu.datasets.utils.custom_transforms import PadToSquare

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Dataset Statistics')
parser.add_argument("--dataset_folder_name", default="CIFAR10", type=str)
parser.add_argument('--threshold_dim', default=225, type=int)
parser.add_argument('--threshold_aspect_ratio', default=2.33, type=float)
parser.add_argument('--save_folder', default="/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/", type=str)


def main():

    args = parser.parse_args()

    dataset_folder = args.dataset_folder_name
    dataset = None
    if dataset_folder == "SUN397":
        dataset = SUN397(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/' + dataset_folder + '/data', split=["train", "test"])
        dataset_gpu = SUN397(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/' + dataset_folder + '/data', split=["train", "test"],
                transform=transforms.Compose([ PadToSquare(), transforms.Resize([200, 200]), transforms.ToTensor()]))
    elif dataset_folder == "Selfie-Image-Detection-Dataset":
        dataset = SIDD(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/' + dataset_folder + '/data', split=["train", "test"])
        dataset_gpu = SIDD(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/' + dataset_folder + '/data', split=["train", "test"],
                transform=transforms.Compose([ PadToSquare(), transforms.Resize([200, 200]), transforms.ToTensor()]))
    elif dataset_folder == "OxfordIII-TPet":
        dataset = OxfordIIITPet(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/' + dataset_folder + '/data', split=["train", "test"])
        dataset_gpu = OxfordIIITPet(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/' + dataset_folder + '/data', split=["train", "test"],
            transform=transforms.Compose([ PadToSquare(), transforms.Resize([200, 200]), transforms.ToTensor()]))
    elif dataset_folder == "IIIT-CFW":
        dataset = CFW(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/' + dataset_folder + '/data', split=["train", "test"])
        dataset_gpu = CFW(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/' + dataset_folder + '/data', split=["train", "test"],
            transform=transforms.Compose([ PadToSquare(), transforms.Resize([200, 200]), transforms.ToTensor()]))
    elif dataset_folder == "ArtImages":
        dataset = ArtImages(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/' + dataset_folder + '/data', split=["train", "test"])
        dataset_gpu = ArtImages(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/' + dataset_folder + '/data', split=["train", "test"],
            transform=transforms.Compose([ PadToSquare(), transforms.Resize([200, 200]), transforms.ToTensor()]))
    elif dataset_folder == "iCartoonFace":
        dataset = iCartoonFace(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/' + dataset_folder + '/data', split=["train", "test"])
        dataset_gpu = iCartoonFace(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/' + dataset_folder + '/data', split=["train", "test"],
                transform=transforms.Compose([ PadToSquare(), transforms.Resize([200, 200]), transforms.ToTensor()]))
    elif dataset_folder == "EMOTIC":
        dataset = EMOTIC(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/' + dataset_folder + '/data', split=["train", "test"])
        dataset_gpu = EMOTIC(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/' + dataset_folder + '/data', split=["train", "test"],
                transform=transforms.Compose([ PadToSquare(), transforms.Resize([200, 200]), transforms.ToTensor()]))


    images_stats_folder = os.path.join(args.save_folder,'statistics', dataset_folder, 'images_stats')
    images_examples_folder = os.path.join(args.save_folder,'statistics', dataset_folder, 'images_examples')
    
    if not os.path.isdir(images_stats_folder):
        os.mkdir(images_stats_folder)
    if os.path.isdir(images_examples_folder):
        shutil.rmtree(images_examples_folder)
    
    os.mkdir(images_examples_folder)
    
    threshold_dim = args.threshold_dim
    
    threshold_aspect_ratio = args.threshold_aspect_ratio
    threshold_area = threshold_dim * threshold_dim * 1/threshold_aspect_ratio

    min_area = 1000000000
    max_area = 0
    areas = []

    min_aspect_ratio = 1000
    max_aspect_ratio = 0
    aspect_ratios = []

    min_H, min_H_W, min_W, min_W_H = 1000, 1000, 1000, 1000
    min_H_img, min_W_img = None, None
    max_H, max_W = 0, 0
    max_W_img_path, max_H_img_path = "", ""
    heights = []
    widths = []


    min_H_img_path, min_W_img_path = "", "" 
    metadata = dataset.metadata

    classes_splits_counter = {}
    for class_name in dataset.classes:
        classes_splits_counter[class_name] = { "train": 0, "test": 0}


    for i, (img, target) in enumerate(dataset):
        W, H = img.size

        area = W*H
        if area > max_area:
            max_area = area
        if area < min_area:
            min_area = area
        areas.append(area)

        aspect_ratio = W/H
        if aspect_ratio > max_aspect_ratio:
            max_aspect_ratio = aspect_ratio
        if aspect_ratio < min_aspect_ratio:
            min_aspect_ratio = aspect_ratio
        aspect_ratios.append(aspect_ratio)


        if H < min_H:
            min_H = H
            min_H_W = W
            min_H_img = img
            min_H_img_path = metadata[i]['img_folder'] + "/" + metadata[i]['img_name']
            
        if H > max_H:
            max_H = H
            max_H_img_path = metadata[i]['img_folder'] + "/" + metadata[i]['img_name']

        if W < min_W:
            min_W = W
            min_W_H = H
            min_W_img = img
            min_W_img_path = metadata[i]['img_folder'] + "/" + metadata[i]['img_name']
            
        if W > max_W:
            max_W = W
            max_W_img_path = metadata[i]['img_folder'] + "/" + metadata[i]['img_name']

        
        heights.append(H)
        widths.append(W)

        classes_splits_counter[target['level3']][metadata[i]["split"]] += 1

    print("The dataset has {} images".format(len(dataset)))
    print("Classes have the following images distribution: \n {}".format(dataset.classes_count))
    print("Classes have been splitted as follows: \n {}".format(classes_splits_counter))


    print("Min height is {}, max height is {}. With height threshold of {}".format(min_H, max_H, threshold_dim,))
    plt.hist(heights, bins="doane", range=[min_H, max_H])
    plt.gca().set(title='Heights Distribution', ylabel='# of images')
    plt.axvline(threshold_dim, color='red')
    plt.text(x=threshold_dim, y=50, s="threshold = {}".format(threshold_dim),)
    plt.savefig(os.path.join(images_stats_folder,"heights_distribution.png"))
    plt.close()

    plt.hist(heights, bins="doane", range=[min_H, threshold_dim*2])
    plt.gca().set(title='Heights Distribution Near Threshold', ylabel='# of images')
    plt.axvline(threshold_dim, color='red')
    plt.text(x=threshold_dim, y=50, s="threshold = {}".format(threshold_dim),)
    plt.savefig(os.path.join(images_stats_folder,"heights_near_threshold_distribution.png"))
    plt.close()

    print("Min width is {}, max width is {}. With width threshold of {}".format(min_W, max_W, threshold_dim,))
    plt.hist(widths, bins="doane", range=[min_W, max_W])
    plt.gca().set(title='Widths Distribution', ylabel='# of images')
    plt.axvline(threshold_dim, color='red')
    plt.text(x=threshold_dim, y=50, s="threshold = {}".format(threshold_dim),)
    plt.savefig(os.path.join(images_stats_folder,"widths_distribution.png"))
    plt.close()

    plt.hist(widths, bins="doane", range=[min_W, threshold_dim*2])
    plt.gca().set(title='Widths Distribution Near Threshold', ylabel='# of images')
    plt.axvline(threshold_dim, color='red')
    plt.text(x=threshold_dim, y=50, s="threshold = {}".format(threshold_dim),)
    plt.savefig(os.path.join(images_stats_folder,"widths_near_threshold_distribution.png"))
    plt.close()
    print("The two most smallest images are: \n with the smallest H: {} W= {} H= {} \n with the smallest W: {} W= {} H= {}"
        .format(min_H_img_path, min_H_W, min_H, min_W_img_path, min_W, min_W_H ))
    
    fig, ax = plt.subplots()
    ax.imshow(min_W_img)
    plt.savefig(os.path.join(images_stats_folder,"image_smallestW.jpg"))
    plt.close()
    
    fig, ax = plt.subplots()
    ax.imshow(min_H_img)
    plt.savefig(os.path.join(images_stats_folder,"image_smallestH.jpg"))
    plt.close()
    

    filtered_imgs = 0
    for area in areas:
        if area < threshold_area:
            filtered_imgs += 1
    print("Min area is {}, max area is {}. With area threshold of {}x{}={} there will be filtered {} images".format(min_area, max_area, threshold_dim, int(threshold_dim*1/threshold_aspect_ratio), int(threshold_area), filtered_imgs))
    plt.hist(areas, bins="doane", range=[min_area, max_area])
    plt.gca().set(title='Areas Distribution', ylabel='# of images')
    plt.axvline(threshold_area, color='red')
    plt.text(x=threshold_area, y=50, s="threshold {}".format(int(threshold_area)),)
    plt.savefig(os.path.join(images_stats_folder,"areas_distribution.png"))
    plt.close()

    plt.hist(areas, bins="doane", range=[min_area, max(threshold_dim*threshold_dim, min_area + 1000)])
    plt.gca().set(title='Areas Distribution Near Threshold', ylabel='# of images')
    plt.axvline(threshold_area, color='red')
    plt.text(x=threshold_area, y=50, s="threshold = {}".format(int(threshold_area)),)
    plt.savefig(os.path.join(images_stats_folder,"areas_near_threshold_distribution.png"))
    plt.close()

    filtered_imgs = 0
    for aspect_ratio in aspect_ratios:
        if aspect_ratio > threshold_aspect_ratio or aspect_ratio < 1/threshold_aspect_ratio:
            filtered_imgs += 1
    print("Min aspect ratio is {}, max aspect ratio is {}. With aspect ratio threshold of {} and his reciprocal {} there will be filtered {} images".format(min_aspect_ratio, max_aspect_ratio, round(threshold_aspect_ratio, 2), round(1/threshold_aspect_ratio, 2), filtered_imgs))
    plt.hist(aspect_ratios, bins="doane", range=[min_aspect_ratio, max_aspect_ratio])
    plt.gca().set(title='Aspect Ratios Distribution', ylabel='# of images')
    plt.axvline(threshold_aspect_ratio, color='red')
    plt.text(x=threshold_aspect_ratio, y=50, s="threshold = {}".format(round(threshold_aspect_ratio, 2)),)
    plt.axvline(1/threshold_aspect_ratio, color='red')
    plt.text(x=1/threshold_aspect_ratio, y=60, s="threshold = {}".format(round(1/threshold_aspect_ratio, 2)),)
    plt.savefig(os.path.join(images_stats_folder,"aspect_ratios_distribution.png"))
    plt.close()

    plt.hist(aspect_ratios, bins="doane", range=[min_aspect_ratio, threshold_aspect_ratio + 0.5])
    plt.gca().set(title='Aspect Ratios Distribution Near Threshold', ylabel='# of images')
    plt.axvline(threshold_aspect_ratio, color='red')
    plt.text(x=threshold_aspect_ratio, y=50, s="threshold = {}".format(round(threshold_aspect_ratio, 2)),)
    plt.axvline(1/threshold_aspect_ratio, color='red')
    plt.text(x=1/threshold_aspect_ratio, y=60, s="threshold = {}".format(round(1/threshold_aspect_ratio, 2)),)
    plt.savefig(os.path.join(images_stats_folder,"aspect_ratios_near_threshold_distribution.png"))
    plt.close()

    classes_count_sorted = {}
    for key, value in sorted(dataset.classes_count.items(), key=lambda item: item[1]):
        classes_count_sorted[key] = value
    plt.xticks(rotation=45, ha='right')
    plt.bar(list(classes_count_sorted.keys())[:30], list(classes_count_sorted.values())[:30], 1.0)    
    plt.gca().set(title='Classes Images Distribution', ylabel='# of images')
    plt.tight_layout()
    plt.savefig(os.path.join(images_stats_folder,"classes_images_distribution.png"))
    plt.close()

    loader = torch.utils.data.DataLoader(dataset_gpu, batch_size=256, shuffle=False, num_workers=1, 
            pin_memory=True, drop_last=False)

    size = 0
    # random list to print some images for debug
    randomlist = []
    for i in range(0,20):
        n = random.randint(1,len(loader))
        randomlist.append(n)
    randomlist[0] = 0

    classes = dataset.classes
    metadata = dataset.metadata


    for i, (img, target) in enumerate(loader):
        img, target = img.cuda(), target
        if i in randomlist:
            # print(metadata[size]['img_folder'])
            # exit()
            print("Image #{} is a {} from split: {} file path: {}/{}".format(size, target['level3'][0],  metadata[size]['split'], metadata[size]['img_folder'], metadata[size]['img_name'] ))
            fig, ax = plt.subplots()
            ax.imshow(img[0].cpu().numpy().transpose([1, 2, 0]))
            labels_text = target['level3'][0]
            plt.gcf().text(0.1+(0.4), 0.02, labels_text, fontsize=8)
            plt.savefig(os.path.join(images_examples_folder, "image_{}.jpg".format(size-1)))
            plt.close()
        size += img.size()[0]

        print("[{}]/[{}] batch iteration".format(i, len(loader)))

if __name__ == '__main__':
    main()


