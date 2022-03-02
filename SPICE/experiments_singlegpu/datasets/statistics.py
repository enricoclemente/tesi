#!/usr/bin/env python
import sys
import argparse
import random
sys.path.insert(0, './')

import torch
from experiments_singlegpu.datasets.SUN397_custom import SUN397
import torchvision.transforms as transforms
from experiments_singlegpu.datasets.custom_transforms import PadToSquare

import matplotlib.pyplot as plt
import matplotlib.patches as patches


parser = argparse.ArgumentParser(description='Dataset Statistics')
parser.add_argument("--dataset_folder_name", default="CIFAR10", type=str)
parser.add_argument('--threshold_dim', default=225, type=int)
parser.add_argument('--threshold_aspect_ratio', default=2.33, type=float)


def main():

    args = parser.parse_args()

    dataset_folder = args.dataset_folder_name
    threshold_dim = args.threshold_dim
    
    threshold_aspect_ratio = args.threshold_aspect_ratio
    threshold_area = threshold_dim * threshold_dim * 1/threshold_aspect_ratio

    dataset = SUN397(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/' + dataset_folder + '/data', split=["train", "test"])

    min_area = 1000000000
    max_area = 0
    areas = []

    min_aspect_ratio = 1000
    max_aspect_ratio = 0
    aspect_ratios = []

    min_H, min_H_W, min_W, min_W_H = 1000, 1000, 1000, 1000
    max_H, max_W = 0, 0
    max_W_img, max_H_img = "", ""
    heights = []
    widths = []


    smallestH, smallestW = "", "" 
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
            smallestH = metadata[i]['img_folder'] + "/" + metadata[i]['img_name']
            fig, ax = plt.subplots()
            ax.imshow((img))
            labels_text = "Image with smallest height is a {}, file path: {}, W= {} H= {}".format(target['level1'], smallestH, min_H_W, min_H)
            plt.gcf().text(0.1, 0.02, labels_text, fontsize=8)
            plt.savefig("experiments_singlegpu/datasets/" + dataset_folder + "/explore_annotations_examples/image_smallestH.jpg")
            plt.close()
        if H > max_H:
            max_H = H
            max_H_img = metadata[i]['img_folder'] + "/" + metadata[i]['img_name']

        if W < min_W:
            min_W = W
            min_W_H = H
            smallestW = metadata[i]['img_folder'] + "/" + metadata[i]['img_name']
            fig, ax = plt.subplots()
            ax.imshow(img)
            labels_text = "Image with smallest width is a {}, file path: {}, W= {} H= {}".format(target['level1'], smallestW, min_W, min_W_H)
            plt.gcf().text(0.1, 0.02, labels_text, fontsize=8)
            plt.savefig("experiments_singlegpu/datasets/" + dataset_folder + "/explore_annotations_examples/image_smallestW.jpg")
            plt.close()

        if W > max_W:
            max_W = W
            max_W_img = metadata[i]['img_folder'] + "/" + metadata[i]['img_name']

        
        heights.append(H)
        widths.append(W)

        classes_splits_counter[target['level1']][metadata[i]["split"]] += 1


    print("The two most smaller images are: \n with the smallest H: {} W= {} H= {} \n with the smallest W: {} W= {} H= {}"
        .format(smallestH, min_H_W, min_H, smallestW, min_W, min_W_H ))

    print("The classes have been splitted in the following numbers: \n {}".format(classes_splits_counter))
        

    plt.hist(areas, bins="doane", range=[min_area, max_area])
    plt.gca().set(title='Areas Distribution', ylabel='# of images')
    plt.axvline(threshold_area, color='red')
    plt.savefig("experiments_singlegpu/datasets/" + dataset_folder + "/explore_annotations_examples/areas_distribution.png")
    plt.close()

    plt.hist(areas, bins="doane", range=[min_area, threshold_dim*threshold_dim])
    plt.gca().set(title='Areas Distribution Near Threshold', ylabel='# of images')
    plt.axvline(threshold_area, color='red')
    plt.savefig("experiments_singlegpu/datasets/" + dataset_folder + "/explore_annotations_examples/areas_near_threshold_distribution.png")
    plt.close()

    exit()
    plt.hist(aspect_ratios, bins="doane", range=[min_aspect_ratio, max_aspect_ratio])
    plt.gca().set(title='Aspect Ratios Distribution', ylabel='# of images')
    plt.savefig("experiments_singlegpu/datasets/" + dataset_folder + "/explore_annotations_examples/aspect_ratios_distribution.png")
    plt.close()

    plt.hist(heights, bins="doane", range=[min_H, max_H])
    plt.gca().set(title='Heights Distribution', ylabel='# of images')
    plt.savefig("experiments_singlegpu/datasets/" + dataset_folder + "/explore_annotations_examples/heights_distribution.png")
    plt.close()

    plt.hist(widths, bins="doane", range=[min_W, max_W])
    plt.gca().set(title='Widths Distribution', ylabel='# of images')
    plt.savefig("experiments_singlegpu/datasets/" + dataset_folder + "/explore_annotations_examples/widths_distribution.png")
    plt.close()

    # print(areas)
    exit()

    dataset = SUN397(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/SUN397/data', split=["train", "test"],
                    transform=transforms.Compose([ PadToSquare(), transforms.Resize([200, 200]), transforms.ToTensor()]))

    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=1, 
            pin_memory=True, drop_last=False)

    size = 0
    print("dataset is big: {}".format(len(dataset)))

    # random list to print some images for debug
    randomlist = []
    for i in range(0,20):
        n = random.randint(1,len(loader))
        randomlist.append(n)
    randomlist[0] = 0

    classes = dataset.classes
    metadata = dataset.metadata

    for i, (img, target) in enumerate(loader):
        img, target = img.cuda(), target.cuda()
        if i in randomlist:
            print("Image #{} is a {} from split: {} file path: {}/{}".format(size, classes[target[0]],  metadata[size]['split'], metadata[size]['img_folder'], metadata[size]['img_name'] ))
            fig, ax = plt.subplots()
            ax.imshow(img[0].cpu().numpy().transpose([1, 2, 0]))
            labels_text = classes[target[0]]
            plt.gcf().text(0.1+(0.4), 0.02, labels_text, fontsize=8)
            plt.savefig("experiments_singlegpu/datasets/SUN397/explore_annotations_examples/sun_image_{}.jpg".format(size-1))
            plt.close()
        size += img.size()[0]

        print("[{}]/[{}] batch iteration".format(i, len(loader)))

    print("The two most smallest images dimensions are: \n {}x{} \n {}x{}"
            .format(loader.dataset.smallest_H_relative_W, loader.dataset.smallest_H, loader.dataset.smallest_W, loader.dataset.smallest_W_relative_H))


if __name__ == '__main__':
    main()


