#!/usr/bin/env python
import sys
import random
import os
import csv
import shutil
import argparse
sys.path.insert(0, './')

import torch
from experiments_singlegpu.datasets.SocialProfilePictures import SocialProfilePictures

import torchvision.transforms as transforms
from experiments_singlegpu.datasets.utils.custom_transforms import PadToSquare

import matplotlib.pyplot as plt

from experiments_singlegpu.datasets.SelfieImageDetectionDataset_custom import SIDD
from experiments_singlegpu.datasets.SUN397_custom import SUN397
from experiments_singlegpu.datasets.OxfordIIITPet_custom import OxfordIIITPet
from experiments_singlegpu.datasets.IIITCFW_custom import CFW
from experiments_singlegpu.datasets.ArtImages_custom import ArtImages
from experiments_singlegpu.datasets.iCartoonFace_custom import iCartoonFace
from experiments_singlegpu.datasets.EMOTIC_custom import EMOTIC

parser = argparse.ArgumentParser(description='SPP metadata generator')
parser.add_argument("--datasets_root_folder", default="/datasets/", metavar="FILE",
                    help="path to save metadata", type=str)
parser.add_argument("--save_folder", default="./dataset/metadata", metavar="FILE",
                    help="path to save metadata", type=str)
parser.add_argument("--version", default="v01", type=str,
                    help='which version of the dataset')


def main():
    args = parser.parse_args()
    if not os.path.isdir(args.save_folder):
        os.mkdir(args.save_folder)

    print("Creating final dataset metadata")
    if args.version == "v01":
        # TODO add datasets_root to this function
        create_images_csv_version_01(csv_save_path=args.save_folder)
    elif args.version == "v02":
        create_images_csv_version_02(datasets_root_folder=args.datasets_root_folder, 
                                    csv_save_path=args.save_folder,
                                    save_folder=args.save_folder)
    elif args.version == "v03":
        create_images_csv_version_03(total_images=80000, datasets_root_folder=args.datasets_root_folder,
                                    csv_save_path=args.save_folder,
                                    save_folder=args.save_folder)

    print("\nCalculating statistics of SocialProfilePictures")
    dataset = SocialProfilePictures(root=args.datasets_root_folder, split=['train','test', 'val'],
                    transform=transforms.Compose([ PadToSquare(), transforms.Resize([225, 225]), transforms.ToTensor()]))

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1, 
                pin_memory=True, drop_last=False)


    print("Dataset is big: {} images".format(len(dataset)))
    print("Target Classes Map: {}".format(dataset.classes_map))
    print("Target Classes Count: {}".format(dataset.classes_count))

    images_examples_folder = os.path.join(args.save_folder, 'images_examples')
    if os.path.isdir(images_examples_folder):
        shutil.rmtree(images_examples_folder)
    os.mkdir(images_examples_folder)


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
            print("Image #{} is a {} from split: {} file path: {}/{}".format(size, metadata[size]['target']['target_level'],  metadata[size]['split'], metadata[size]['img_folder'], metadata[size]['img_name'] ))
            fig, ax = plt.subplots()
            ax.imshow(img[0].cpu().numpy().transpose([1, 2, 0]))
            labels_text = classes[target[0]]
            plt.gcf().text(0.1+(0.4), 0.02, labels_text, fontsize=8)
            plt.savefig(os.path.join(images_examples_folder, "image_{}.jpg".format(size-1)))
            plt.close()
        size += img.size()[0]

        print("[{}]/[{}] batch iteration".format(i, len(loader)))


"""
    Create Images CSV of the final dataset starting from the datasets
    Args:
        - total_images: the number of total images of the final dataset
        - split_perc: the percetange of split between train images and test images
        - aspect_ratio_threshold: floating value indicating the trheshold to filter images that are 
                                    over a certain aspect ratio
        - dim_threshold: integer value indicating min dimension threshold to filter images that have area
                        smaller than dim_threshold * dim_threshold * 1/aspect_ratio_threshold
                        example: 
                            aspect_ratio_threshold = 2.33 (21:9)
                            dim_threshold = 225 (px) means images that have area small than 225x96(dim_threshold * 1/aspect_ratio_threshold)
        - csv_save_path: string indicating path where to save images_metadata.csv
    
    The images in the classes will keep the same proportions as from the original statistics
"""
def create_images_csv_version_01(total_images=60000, split_perc=0.8, aspect_ratio_threshold=2.33, dim_threshold=225, csv_save_path='./'):
    # indicates level0 classes and relative percentual presence in the final dataset
    level0_classes = {  'people': { 'perc': 0.6 },
                        'scenes': { 'perc': 0.2 },
                        'other': { 'perc': 0.2 }}

    dataset_colors = {'SIDD_dataset_filtered_partition': 'blue', 
                    'SUN397_dataset_filtered_partition': 'red', 
                    'Oxford_dataset_filtered_partition': 'green',
                    'Cartoon_dataset_filtered_partition': 'pink', 
                    'ArtImages_dataset_filtered_partition': 'purple'}

    dataset_legend = {'People': 'blue', "Scenes": 'red', "Pets": 'green', "Cartoon": "pink", 'Art': 'purple'}
    # First open the datasets and gets overall statistics

    # Working on level0: people
    print("Level0: people")
    # Take a partition of SIDD dataset == total_images * people perc
    SIDD_dataset_original = SIDD(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/Selfie-Image-Detection-Dataset/data',
                split=["train", "test"], split_perc=split_perc)
    SIDD_dataset_filtered = SIDD(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/Selfie-Image-Detection-Dataset/data',
                    split=["train", "test"], split_perc=split_perc, 
                    aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    SIDD_partition_perc = level0_classes['people']['perc'] * total_images / len(SIDD_dataset_filtered)
    SIDD_dataset_filtered_partition = SIDD(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/Selfie-Image-Detection-Dataset/data',
                    split=["train", "test"], split_perc=split_perc, partition_perc=SIDD_partition_perc, distribute_images=True,
                    aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    
    print("\tSIDD dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(SIDD_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(SIDD_dataset_filtered), SIDD_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(SIDD_dataset_filtered_partition), SIDD_partition_perc))
    print("\t\tclasses partition: {}".format(SIDD_dataset_filtered_partition.classes_count))
    print("\t\tclasses filtering ratio: {}".format(SIDD_dataset_filtered_partition.filtering_classes_effect))

    # Working on level0: scenes
    print("Level0: scenes")
    SUN397_dataset_original = SUN397(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/SUN397/data', 
                    split=["train","test"], split_perc=split_perc)
    SUN397_dataset_filtered = SUN397(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/SUN397/data', 
                        split=["train","test"], split_perc=split_perc, 
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    SUN397_partition_perc = level0_classes['scenes']['perc'] * total_images / len(SUN397_dataset_filtered)
    SUN397_dataset_filtered_partition = SUN397(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/SUN397/data', 
                        split=["train","test"], split_perc=split_perc, partition_perc=SUN397_partition_perc, distribute_images=True, distribute_level='level2',
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    
    
    print("\tSUN397 dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(SUN397_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(SUN397_dataset_filtered), SUN397_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(SUN397_dataset_filtered_partition), SUN397_partition_perc))
    print("\t\tclasses partition: {}".format(SUN397_dataset_filtered_partition.classes_count))
    print("\t\tscenes classes hierarchy: {}".format(SUN397_dataset_filtered_partition.classes_hierarchy))
    print("\t\tclasses filtering ratio: {}".format(SUN397_dataset_filtered_partition.filtering_classes_effect))
    # SUN397_total_ver = 0
    # for count in SUN397_dataset.classes_count.values():
    #     SUN397_total_ver += count
    # print(SUN397_total_ver)
    
    print("Level0: other")
    print("Level1: Pets")
    Oxford_dataset_original = OxfordIIITPet(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/OxfordIII-TPet/data', 
                    split=["train","test"], split_perc=split_perc)
    Oxford_dataset_filtered = OxfordIIITPet(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/OxfordIII-TPet/data', 
                        split=["train","test"], split_perc=split_perc, 
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    Oxford_partition_perc = level0_classes['other']['perc']/3 * total_images / len(Oxford_dataset_filtered)
    Oxford_dataset_filtered_partition = OxfordIIITPet(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/OxfordIII-TPet/data', 
                        split=["train","test"], split_perc=split_perc, partition_perc=Oxford_partition_perc, distribute_images=True, distribute_level='level2',
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    print("\tOxfordIII-Tpet dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(Oxford_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(Oxford_dataset_filtered), Oxford_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(Oxford_dataset_filtered_partition), Oxford_partition_perc))
    print("\t\tclasses partition: {}".format(Oxford_dataset_filtered_partition.classes_count))
    print("\t\tscenes classes hierarchy: {}".format(Oxford_dataset_filtered_partition.classes_hierarchy))
    print("\t\tclasses filtering ratio: {}".format(Oxford_dataset_filtered_partition.filtering_classes_effect))
    
    print("Level1: Cartoon")
    Cartoon_dataset_original = iCartoonFace(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/iCartoonFace/data', 
                    split=["train","test"], split_perc=split_perc)
    Cartoon_dataset_filtered = iCartoonFace(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/iCartoonFace/data', 
                        split=["train","test"], split_perc=split_perc, 
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    Cartoon_partition_perc = level0_classes['other']['perc']/3 * total_images / len(Cartoon_dataset_filtered)
    Cartoon_dataset_filtered_partition = iCartoonFace(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/iCartoonFace/data', 
                        split=["train","test"], split_perc=split_perc, partition_perc=Cartoon_partition_perc, distribute_images=True,
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    print("\tiCartoonFace dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(Cartoon_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(Cartoon_dataset_filtered), Cartoon_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(Cartoon_dataset_filtered_partition), Cartoon_partition_perc))
    print("\t\tclasses partition: {}".format(Cartoon_dataset_filtered_partition.classes_count))
    print("\t\tclasses filtering ratio: {}".format(Cartoon_dataset_filtered_partition.filtering_classes_effect))
    # CFW_dataset_original = CFW(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/IIIT-CFW/data', 
    #                 split=["train","test"], split_perc=split_perc)
    # CFW_dataset_filtered = CFW(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/IIIT-CFW/data', 
    #                     split=["train","test"], split_perc=split_perc, 
    #                     aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    # CFW_partition_perc = level0_classes['other']['perc']/3 * total_images / len(CFW_dataset_filtered)
    # if CFW_partition_perc > 1.0:
    #     print("CFW database has not enough images!")
    #     CFW_partition_perc = 0.99
    # CFW_dataset_filtered_partition = CFW(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/IIIT-CFW/data', 
    #                     split=["train","test"], split_perc=split_perc, partition_perc=CFW_partition_perc, distribute_images=True,
    #                     aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    # print("\tIIITCFW dataset statistics")
    # print("\t\toriginal dataset images: {}".format(len(CFW_dataset_original)))
    # print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(CFW_dataset_filtered), CFW_dataset_filtered.total_filtered))
    # print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(CFW_dataset_filtered_partition), CFW_partition_perc))
    # print("\t\tclasses partition: {}".format(CFW_dataset_filtered_partition.classes_count))
    # print("\t\tclasses filtering ratio: {}".format(CFW_dataset_filtered_partition.filtering_classes_effect))
    

    print("Level1: Art")
    ArtImages_dataset_original = ArtImages(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/ArtImages/data', 
                    split=["train","test"], split_perc=split_perc)
    ArtImages_dataset_filtered = ArtImages(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/ArtImages/data', 
                        split=["train","test"], split_perc=split_perc, 
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    ArtImages_partition_perc = level0_classes['other']['perc']/3 * total_images / len(ArtImages_dataset_filtered)
    ArtImages_dataset_filtered_partition = ArtImages(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/ArtImages/data', 
                        split=["train","test"], split_perc=split_perc, partition_perc=ArtImages_partition_perc, distribute_images=True,
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    print("\tArtImages dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(ArtImages_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(ArtImages_dataset_filtered), ArtImages_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(ArtImages_dataset_filtered_partition), ArtImages_partition_perc))
    print("\t\tclasses partition: {}".format(ArtImages_dataset_filtered_partition.classes_count))
    print("\t\tclasses filtering ratio: {}".format(ArtImages_dataset_filtered_partition.filtering_classes_effect))

   
    final_dataset_level_1_classes_count = {}
    final_dataset_level_2_classes_count = {}
    final_dataset_level_3_classes_count = {}
    for d in [SIDD_dataset_filtered_partition, SUN397_dataset_filtered_partition, Oxford_dataset_filtered_partition,
                Cartoon_dataset_filtered_partition, ArtImages_dataset_filtered_partition]:
        for t in d.targets:
            if t['level1'] not in final_dataset_level_1_classes_count:
                final_dataset_level_1_classes_count[t['level1']] = 1
            elif t['level1'] in final_dataset_level_1_classes_count:
                final_dataset_level_1_classes_count[t['level1']] += 1

            if t['level2'] not in final_dataset_level_2_classes_count:
                final_dataset_level_2_classes_count[t['level2']] = 1
            elif t['level2'] in final_dataset_level_2_classes_count:
                final_dataset_level_2_classes_count[t['level2']] += 1

            if t['level3'] not in final_dataset_level_3_classes_count:
                final_dataset_level_3_classes_count[t['level3']] = 1
            elif t['level3'] in final_dataset_level_3_classes_count:
                final_dataset_level_3_classes_count[t['level3']] += 1

    hist_elements = 30
    # print(final_dataset_level_2_classes_count)    
    final_dataset_level_1_classes_count_sorted = {}
    final_dataset_level_1_classes_color = []
    for key, value in sorted(final_dataset_level_1_classes_count.items(), key=lambda item: item[1]):
        final_dataset_level_1_classes_count_sorted[key] = value
        for i, d in enumerate([SIDD_dataset_filtered_partition, SUN397_dataset_filtered_partition, Oxford_dataset_filtered_partition,
                Cartoon_dataset_filtered_partition, ArtImages_dataset_filtered_partition]):
            if key in d.classes_map['level1']:
                final_dataset_level_1_classes_color.append(list(dataset_colors.values())[i])

    print("Level 1 Classes Distribution: {}".format(final_dataset_level_1_classes_count_sorted))
    plt.xticks(rotation=45, ha='right')
    plt.bar(list(final_dataset_level_1_classes_count_sorted.keys()), list(final_dataset_level_1_classes_count_sorted.values()), color=final_dataset_level_1_classes_color)
    plt.gca().set(title='Level 1 Partitioned Filtered Classes Distribution', ylabel='# of images')
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(list(final_dataset_level_1_classes_count_sorted.values())):
        plt.text(xlocs[i] - 0.3, v + 0.1, str(v), fontsize=8)
    for i, j in dataset_legend.items(): #Loop over color dictionary
        plt.bar(i, 0,width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("experiments_singlegpu/datasets/final_dataset_filtered_partition_level1_classes_distribution.svg")
    plt.close()

    final_dataset_level_2_classes_count_sorted = {}
    final_dataset_level_2_classes_color = []
    for key, value in sorted(final_dataset_level_2_classes_count.items(), key=lambda item: item[1]):
        final_dataset_level_2_classes_count_sorted[key] = value
        for i, d in enumerate([SIDD_dataset_filtered_partition, SUN397_dataset_filtered_partition, Oxford_dataset_filtered_partition,
                Cartoon_dataset_filtered_partition, ArtImages_dataset_filtered_partition]):
            if key in d.classes_map['level2']:
                final_dataset_level_2_classes_color.append(list(dataset_colors.values())[i])
    print("Level 2 Classes Distribution: {}".format(final_dataset_level_2_classes_count_sorted))
    plt.xticks(rotation=45, ha='right')
    plt.bar(list(final_dataset_level_2_classes_count_sorted.keys()), list(final_dataset_level_2_classes_count_sorted.values()), color=final_dataset_level_2_classes_color)
    plt.gca().set(title='Level 2 Partitioned Filtered Classes Distribution', ylabel='# of images')
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(list(final_dataset_level_2_classes_count_sorted.values())):
        plt.text(xlocs[i] - 0.3, v + 0.1, str(v), fontsize=4)
    for i, j in dataset_legend.items(): #Loop over color dictionary
        plt.bar(i, 0,width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("experiments_singlegpu/datasets/final_dataset_filtered_partition_level2_classes_distribution.svg")
    plt.close()
    
    final_dataset_level_3_classes_count_sorted = {}
    final_dataset_level_3_classes_color = []
    for key, value in sorted(final_dataset_level_3_classes_count.items(), key=lambda item: item[1]):
        final_dataset_level_3_classes_count_sorted[key] = value
        for i, d in enumerate([SIDD_dataset_filtered_partition, SUN397_dataset_filtered_partition, Oxford_dataset_filtered_partition,
                Cartoon_dataset_filtered_partition, ArtImages_dataset_filtered_partition]):
            if key in d.classes_map['level3']:
                final_dataset_level_3_classes_color.append(list(dataset_colors.values())[i])
    print("Level 3 Classes Distribution: {}".format(final_dataset_level_3_classes_count_sorted))
    plt.xticks(rotation=45, ha='right')
    plt.bar(list(final_dataset_level_3_classes_count_sorted.keys())[:hist_elements], list(final_dataset_level_3_classes_count_sorted.values())[:hist_elements], color=final_dataset_level_3_classes_color[:hist_elements])
    plt.gca().set(title='Level 3 Partitioned Filtered Classes Distribution', ylabel='# of images')
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(list(final_dataset_level_3_classes_count_sorted.values())[:hist_elements]):
        plt.text(xlocs[i] - 0.3, v + 0.1, str(v), fontsize=4)
    for i, j in dataset_legend.items(): #Loop over color dictionary
        plt.bar(i, 0,width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("experiments_singlegpu/datasets/final_dataset_filtered_partition_level3_classes_distribution.svg")
    plt.close()


    # creation of the images csv
    with open(os.path.join(csv_save_path + 'images_metadata.csv'), mode='w') as file:
        csv_writer = csv.writer(file, delimiter=',')
        csv_writer.writerow(['original_dataset', 'img_folder', 'img_name', 'split', 'level0', 'level1', 'level2', 'level3', 'target_level'])

        for m in SIDD_dataset_filtered_partition.metadata:
            csv_writer.writerow(['Selfie-Image-Detection-Dataset', m['img_folder'], m['img_name'], m['split'], 'people', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level3']])
        for m in SUN397_dataset_filtered_partition.metadata:
            csv_writer.writerow(['SUN397', m['img_folder'], m['img_name'], m['split'], 'scenes', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level2']])
        for m in Oxford_dataset_filtered_partition.metadata:
            csv_writer.writerow(['OxfordIII-TPet', m['img_folder'], m['img_name'], m['split'], 'other', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level2']])
        for m in Cartoon_dataset_filtered_partition.metadata:
            csv_writer.writerow(['iCartoonFace', m['img_folder'], m['img_name'], m['split'], 'other', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level3']])
        for m in ArtImages_dataset_filtered_partition.metadata:
            csv_writer.writerow(['ArtImages', m['img_folder'], m['img_name'], m['split'], 'other', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level3']])

"""
    In this version nonselfie images from Selfie-Image-Detection-Dataset are replaced with EMOTIC images since the 
    Selfie-Image-Detection-Dataset nonselfie images may be images with no people at all
"""
def create_images_csv_version_02(total_images=60000, split_perc=0.8, aspect_ratio_threshold=2.33, dim_threshold=225, datasets_root_folder='/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets/', csv_save_path='./', save_folder='./'):
    # indicates level0 classes and relative percentual presence in the final dataset
    level0_classes = {  'people': { 'perc': 0.6 },
                        'scenes': { 'perc': 0.2 },
                        'other': { 'perc': 0.2 }}

    dataset_colors = {'SIDD_dataset_filtered_partition': 'blue',
                    'EMOTIC_dataset_filtered_partition': 'blue', 
                    'SUN397_dataset_filtered_partition': 'red', 
                    'Oxford_dataset_filtered_partition': 'green',
                    'Cartoon_dataset_filtered_partition': 'pink', 
                    'ArtImages_dataset_filtered_partition': 'purple'}

    dataset_legend = {'People': 'blue', "Scenes": 'red', "Pets": 'green', "Cartoon": "pink", 'Art': 'purple'}
    # First open the datasets and gets overall statistics

    datasets_references = []
    # Working on level0: people
    print("Level0: people")
    # Take a partition of SIDD dataset == total_images * people perc
    SIDD_dataset_original = SIDD(root=os.path.join(datasets_root_folder, 'Selfie-Image-Detection-Dataset/data'),
                split=["train", "test"], split_perc=split_perc)
    SIDD_dataset_filtered = SIDD(root=os.path.join(datasets_root_folder, 'Selfie-Image-Detection-Dataset/data'),
                    split=["train", "test"], split_perc=split_perc, 
                    aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    SIDD_partition_perc = min(level0_classes['people']['perc'] * total_images / len(SIDD_dataset_filtered), 1.0)
    SIDD_dataset_filtered_partition = SIDD(root=os.path.join(datasets_root_folder, 'Selfie-Image-Detection-Dataset/data'),
                    split=["train", "test"], split_perc=split_perc, partition_perc=SIDD_partition_perc, distribute_images=True,
                    aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    datasets_references.append(SIDD_dataset_filtered_partition)
    
    print("\tSIDD dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(SIDD_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(SIDD_dataset_filtered), SIDD_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(SIDD_dataset_filtered_partition), SIDD_partition_perc))
    print("\t\tclasses partition: {}".format(SIDD_dataset_filtered_partition.classes_count))
    print("\t\tclasses filtering ratio: {}".format(SIDD_dataset_filtered_partition.filtering_classes_effect))
    
    # Take a partition of SIDD dataset == total_images * people perc
    EMOTIC_dataset_original = EMOTIC(root=os.path.join(datasets_root_folder, 'EMOTIC/data'),
                split=["train", "test"], split_perc=split_perc)
    EMOTIC_dataset_filtered = EMOTIC(root=os.path.join(datasets_root_folder, 'EMOTIC/data'),
                    split=["train", "test"], split_perc=split_perc, 
                    aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    EMOTIC_partition_perc = min(level0_classes['people']['perc']/2 * total_images / len(EMOTIC_dataset_filtered), 1.0)
    EMOTIC_dataset_filtered_partition = EMOTIC(root=os.path.join(datasets_root_folder, 'EMOTIC/data'),
                    split=["train", "test"], split_perc=split_perc, partition_perc=EMOTIC_partition_perc, distribute_images=True,
                    aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    datasets_references.append(EMOTIC_dataset_filtered_partition)

    print("\tEMOTIC dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(EMOTIC_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(EMOTIC_dataset_filtered), EMOTIC_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(EMOTIC_dataset_filtered_partition), EMOTIC_partition_perc))
    print("\t\tclasses partition: {}".format(EMOTIC_dataset_filtered_partition.classes_count))
    print("\t\tclasses filtering ratio: {}".format(EMOTIC_dataset_filtered_partition.filtering_classes_effect))

    # Working on level0: scenes
    print("Level0: scenes")
    SUN397_dataset_original = SUN397(root=os.path.join(datasets_root_folder, 'SUN397/data'), 
                    split=["train","test"], split_perc=split_perc)
    SUN397_dataset_filtered = SUN397(root=os.path.join(datasets_root_folder, 'SUN397/data'), 
                        split=["train","test"], split_perc=split_perc, 
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    SUN397_partition_perc = min(level0_classes['scenes']['perc'] * total_images / len(SUN397_dataset_filtered), 1.0)
    SUN397_dataset_filtered_partition = SUN397(root=os.path.join(datasets_root_folder, 'SUN397/data'), 
                        split=["train","test"], split_perc=split_perc, partition_perc=SUN397_partition_perc, distribute_images=True, distribute_level='level2',
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    datasets_references.append(SUN397_dataset_filtered_partition)
    
    print("\tSUN397 dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(SUN397_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(SUN397_dataset_filtered), SUN397_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(SUN397_dataset_filtered_partition), SUN397_partition_perc))
    print("\t\tclasses partition: {}".format(SUN397_dataset_filtered_partition.classes_count))
    print("\t\tscenes classes hierarchy: {}".format(SUN397_dataset_filtered_partition.classes_hierarchy))
    print("\t\tclasses filtering ratio: {}".format(SUN397_dataset_filtered_partition.filtering_classes_effect))
    
    print("Level0: other")
    print("Level1: Pets")
    Oxford_dataset_original = OxfordIIITPet(root=os.path.join(datasets_root_folder, 'OxfordIII-TPet/data'), 
                    split=["train","test"], split_perc=split_perc)
    Oxford_dataset_filtered = OxfordIIITPet(root=os.path.join(datasets_root_folder, 'OxfordIII-TPet/data'), 
                        split=["train","test"], split_perc=split_perc, 
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    Oxford_partition_perc = min(level0_classes['other']['perc']/3 * total_images / len(Oxford_dataset_filtered), 1.0)
    Oxford_dataset_filtered_partition = OxfordIIITPet(root=os.path.join(datasets_root_folder, 'OxfordIII-TPet/data'), 
                        split=["train","test"], split_perc=split_perc, partition_perc=Oxford_partition_perc, distribute_images=True, distribute_level='level2',
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    datasets_references.append(Oxford_dataset_filtered_partition)

    print("\tOxfordIII-Tpet dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(Oxford_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(Oxford_dataset_filtered), Oxford_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(Oxford_dataset_filtered_partition), Oxford_partition_perc))
    print("\t\tclasses partition: {}".format(Oxford_dataset_filtered_partition.classes_count))
    print("\t\tscenes classes hierarchy: {}".format(Oxford_dataset_filtered_partition.classes_hierarchy))
    print("\t\tclasses filtering ratio: {}".format(Oxford_dataset_filtered_partition.filtering_classes_effect))
    
    print("Level1: Cartoon")
    Cartoon_dataset_original = iCartoonFace(root=os.path.join(datasets_root_folder, 'iCartoonFace/data'), 
                    split=["train","test"], split_perc=split_perc)
    Cartoon_dataset_filtered = iCartoonFace(root=os.path.join(datasets_root_folder, 'iCartoonFace/data'), 
                        split=["train","test"], split_perc=split_perc, 
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    Cartoon_partition_perc = min(level0_classes['other']['perc']/3 * total_images / len(Cartoon_dataset_filtered), 1.0)
    Cartoon_dataset_filtered_partition = iCartoonFace(root=os.path.join(datasets_root_folder, 'iCartoonFace/data'), 
                        split=["train","test"], split_perc=split_perc, partition_perc=Cartoon_partition_perc, distribute_images=True,
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    datasets_references.append(Cartoon_dataset_filtered_partition)

    print("\tiCartoonFace dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(Cartoon_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(Cartoon_dataset_filtered), Cartoon_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(Cartoon_dataset_filtered_partition), Cartoon_partition_perc))
    print("\t\tclasses partition: {}".format(Cartoon_dataset_filtered_partition.classes_count))
    print("\t\tclasses filtering ratio: {}".format(Cartoon_dataset_filtered_partition.filtering_classes_effect))    

    print("Level1: Art")
    ArtImages_dataset_original = ArtImages(root=os.path.join(datasets_root_folder, 'ArtImages/data'), 
                    split=["train","test"], split_perc=split_perc)
    ArtImages_dataset_filtered = ArtImages(root=os.path.join(datasets_root_folder, 'ArtImages/data'), 
                        split=["train","test"], split_perc=split_perc, 
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    ArtImages_partition_perc = min(level0_classes['other']['perc']/3 * total_images / len(ArtImages_dataset_filtered), 1.0)
    ArtImages_dataset_filtered_partition = ArtImages(root=os.path.join(datasets_root_folder, 'ArtImages/data'), 
                        split=["train","test"], split_perc=split_perc, partition_perc=ArtImages_partition_perc, distribute_images=True,
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    datasets_references.append(ArtImages_dataset_filtered_partition)

    print("\tArtImages dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(ArtImages_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(ArtImages_dataset_filtered), ArtImages_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(ArtImages_dataset_filtered_partition), ArtImages_partition_perc))
    print("\t\tclasses partition: {}".format(ArtImages_dataset_filtered_partition.classes_count))
    print("\t\tclasses filtering ratio: {}".format(ArtImages_dataset_filtered_partition.filtering_classes_effect))

   
    # counting overall classes for each level
    final_dataset_level_1_classes_count = {}
    final_dataset_level_2_classes_count = {}
    final_dataset_level_3_classes_count = {}
    for d in datasets_references:
        for t in d.targets:
            if not (d == SIDD_dataset_filtered_partition and (t['level1'] == 'nonselfie' or t['level2'] == 'nonselfie' or t['level3'] == 'nonselfie')): 
                if t['level1'] not in final_dataset_level_1_classes_count:
                    final_dataset_level_1_classes_count[t['level1']] = 1
                elif t['level1'] in final_dataset_level_1_classes_count:
                    final_dataset_level_1_classes_count[t['level1']] += 1

                if t['level2'] not in final_dataset_level_2_classes_count:
                    final_dataset_level_2_classes_count[t['level2']] = 1
                elif t['level2'] in final_dataset_level_2_classes_count:
                    final_dataset_level_2_classes_count[t['level2']] += 1

                if t['level3'] not in final_dataset_level_3_classes_count:
                    final_dataset_level_3_classes_count[t['level3']] = 1
                elif t['level3'] in final_dataset_level_3_classes_count:
                    final_dataset_level_3_classes_count[t['level3']] += 1

    hist_elements = 30
    # print(final_dataset_level_2_classes_count)    
    final_dataset_level_1_classes_count_sorted = {}
    final_dataset_level_1_classes_color = []
    for key, value in sorted(final_dataset_level_1_classes_count.items(), key=lambda item: item[1]):
        final_dataset_level_1_classes_count_sorted[key] = value
        for i, d in enumerate(datasets_references):
            if key in d.classes_map['level1']:
                final_dataset_level_1_classes_color.append(list(dataset_colors.values())[i])

    print("Level 1 Classes Distribution: {}".format(final_dataset_level_1_classes_count_sorted))
    plt.xticks(rotation=45, ha='right')
    plt.bar(list(final_dataset_level_1_classes_count_sorted.keys()), list(final_dataset_level_1_classes_count_sorted.values()), color=final_dataset_level_1_classes_color)
    plt.gca().set(title='Level 1 Partitioned Filtered Classes Distribution', ylabel='# of images')
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(list(final_dataset_level_1_classes_count_sorted.values())):
        plt.text(xlocs[i] - 0.3, v + 0.1, str(v), fontsize=8)
    for i, j in dataset_legend.items(): #Loop over color dictionary
        plt.bar(i, 0,width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("{}/filtered_partition_level1_classes_distribution.svg".format(save_folder))
    plt.close()

    final_dataset_level_2_classes_count_sorted = {}
    final_dataset_level_2_classes_color = []
    for key, value in sorted(final_dataset_level_2_classes_count.items(), key=lambda item: item[1]):
        final_dataset_level_2_classes_count_sorted[key] = value
        for i, d in enumerate(datasets_references):
            if key in d.classes_map['level2']:
                final_dataset_level_2_classes_color.append(list(dataset_colors.values())[i])
    print("Level 2 Classes Distribution: {}".format(final_dataset_level_2_classes_count_sorted))
    plt.xticks(rotation=45, ha='right')
    plt.bar(list(final_dataset_level_2_classes_count_sorted.keys()), list(final_dataset_level_2_classes_count_sorted.values()), color=final_dataset_level_2_classes_color)
    plt.gca().set(title='Level 2 Partitioned Filtered Classes Distribution', ylabel='# of images')
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(list(final_dataset_level_2_classes_count_sorted.values())):
        plt.text(xlocs[i] - 0.3, v + 0.1, str(v), fontsize=4)
    for i, j in dataset_legend.items(): #Loop over color dictionary
        plt.bar(i, 0,width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("{}/filtered_partition_level2_classes_distribution.svg".format(save_folder))
    plt.close()
    
    final_dataset_level_3_classes_count_sorted = {}
    final_dataset_level_3_classes_color = []
    for key, value in sorted(final_dataset_level_3_classes_count.items(), key=lambda item: item[1]):
        final_dataset_level_3_classes_count_sorted[key] = value
        for i, d in enumerate(datasets_references):
            if key in d.classes_map['level3']:
                final_dataset_level_3_classes_color.append(list(dataset_colors.values())[i])
    print("Level 3 Classes Distribution: {}".format(final_dataset_level_3_classes_count_sorted))
    plt.xticks(rotation=45, ha='right')
    plt.bar(list(final_dataset_level_3_classes_count_sorted.keys())[:hist_elements], list(final_dataset_level_3_classes_count_sorted.values())[:hist_elements], color=final_dataset_level_3_classes_color[:hist_elements])
    plt.gca().set(title='Level 3 Partitioned Filtered Classes Distribution', ylabel='# of images')
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(list(final_dataset_level_3_classes_count_sorted.values())[:hist_elements]):
        plt.text(xlocs[i] - 0.3, v + 0.1, str(v), fontsize=4)
    for i, j in dataset_legend.items(): #Loop over color dictionary
        plt.bar(i, 0,width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("{}/filtered_partition_level3_classes_distribution.svg".format(save_folder))
    plt.close()


    # creation of the images csv
    with open(os.path.join(csv_save_path, 'images_metadata.csv'), mode='w') as file:
        csv_writer = csv.writer(file, delimiter=',')
        csv_writer.writerow(['original_dataset', 'img_folder', 'img_name', 'split', 'level0', 'level1', 'level2', 'level3', 'target_level'])

        for m in SIDD_dataset_filtered_partition.metadata:
            if m['target']['level1'] == 'selfie':
                csv_writer.writerow(['Selfie-Image-Detection-Dataset', m['img_folder'], m['img_name'], m['split'], 'people', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level3']])
        for m in EMOTIC_dataset_filtered.metadata:
            csv_writer.writerow(['EMOTIC', m['img_folder'], m['img_name'], m['split'], 'people', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level3']])
        for m in SUN397_dataset_filtered_partition.metadata:
            csv_writer.writerow(['SUN397', m['img_folder'], m['img_name'], m['split'], 'scenes', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level2']])
        for m in Oxford_dataset_filtered_partition.metadata:
            csv_writer.writerow(['OxfordIII-TPet', m['img_folder'], m['img_name'], m['split'], 'other', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level2']])
        for m in Cartoon_dataset_filtered_partition.metadata:
            csv_writer.writerow(['iCartoonFace', m['img_folder'], m['img_name'], m['split'], 'other', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level3']])
        for m in ArtImages_dataset_filtered_partition.metadata:
            csv_writer.writerow(['ArtImages', m['img_folder'], m['img_name'], m['split'], 'other', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level3']])


def create_images_csv_version_03(total_images=60000, split_perc=0.8, aspect_ratio_threshold=2.33, dim_threshold=225, datasets_root_folder='/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets/', csv_save_path='./', save_folder='./'):
    # indicates level0 classes and relative percentual presence in the final dataset
    level0_classes = {  'people': { 'perc': 0.4 },
                        'scenes': { 'perc': 0.3 },
                        'other': { 'perc': 0.3 }}

    dataset_colors = {'SIDD_dataset_filtered_partition': 'blue',
                    'EMOTIC_dataset_filtered_partition': 'blue', 
                    'SUN397_dataset_filtered_partition': 'red', 
                    'Oxford_dataset_filtered_partition': 'green',
                    'Cartoon_dataset_filtered_partition': 'pink', 
                    'ArtImages_dataset_filtered_partition': 'purple'}

    dataset_legend = {'People': 'blue', "Scenes": 'red', "Pets": 'green', "Cartoon": "pink", 'Art': 'purple'}
    # First open the datasets and gets overall statistics

    datasets_references = []
    # Working on level0: people
    print("Level0: people")
    # Take a partition of SIDD dataset == total_images * people perc
    SIDD_dataset_original = SIDD(root=os.path.join(datasets_root_folder, 'Selfie-Image-Detection-Dataset/data'),
                split=["train", "test"], split_perc=split_perc)
    SIDD_dataset_filtered = SIDD(root=os.path.join(datasets_root_folder, 'Selfie-Image-Detection-Dataset/data'),
                    split=["train", "test"], split_perc=split_perc, 
                    aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    SIDD_partition_perc = min(level0_classes['people']['perc'] * total_images / len(SIDD_dataset_filtered), 1.0)
    SIDD_dataset_filtered_partition = SIDD(root=os.path.join(datasets_root_folder, 'Selfie-Image-Detection-Dataset/data'),
                    split=["train", "test"], split_perc=split_perc, partition_perc=SIDD_partition_perc, distribute_images=True,
                    aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    datasets_references.append(SIDD_dataset_filtered_partition)
    
    print("\tSIDD dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(SIDD_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(SIDD_dataset_filtered), SIDD_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(SIDD_dataset_filtered_partition), SIDD_partition_perc))
    print("\t\tclasses partition: {}".format(SIDD_dataset_filtered_partition.classes_count))
    print("\t\tclasses filtering ratio: {}".format(SIDD_dataset_filtered_partition.filtering_classes_effect))
    
    # Take a partition of SIDD dataset == total_images * people perc
    EMOTIC_dataset_original = EMOTIC(root=os.path.join(datasets_root_folder, 'EMOTIC/data'),
                split=["train", "test"], split_perc=split_perc)
    EMOTIC_dataset_filtered = EMOTIC(root=os.path.join(datasets_root_folder, 'EMOTIC/data'),
                    split=["train", "test"], split_perc=split_perc, 
                    aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    EMOTIC_partition_perc = min(level0_classes['people']['perc']/2 * total_images / len(EMOTIC_dataset_filtered), 1.0)
    EMOTIC_dataset_filtered_partition = EMOTIC(root=os.path.join(datasets_root_folder, 'EMOTIC/data'),
                    split=["train", "test"], split_perc=split_perc, partition_perc=EMOTIC_partition_perc, distribute_images=True,
                    aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    datasets_references.append(EMOTIC_dataset_filtered_partition)

    print("\tEMOTIC dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(EMOTIC_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(EMOTIC_dataset_filtered), EMOTIC_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(EMOTIC_dataset_filtered_partition), EMOTIC_partition_perc))
    print("\t\tclasses partition: {}".format(EMOTIC_dataset_filtered_partition.classes_count))
    print("\t\tclasses filtering ratio: {}".format(EMOTIC_dataset_filtered_partition.filtering_classes_effect))

    # Working on level0: scenes
    print("Level0: scenes")
    SUN397_dataset_original = SUN397(root=os.path.join(datasets_root_folder, 'SUN397/data'), 
                    split=["train","test"], split_perc=split_perc)
    SUN397_dataset_filtered = SUN397(root=os.path.join(datasets_root_folder, 'SUN397/data'), 
                        split=["train","test"], split_perc=split_perc, 
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    SUN397_partition_perc = min(level0_classes['scenes']['perc'] * total_images / len(SUN397_dataset_filtered), 1.0)
    SUN397_dataset_filtered_partition = SUN397(root=os.path.join(datasets_root_folder, 'SUN397/data'), 
                        split=["train","test"], split_perc=split_perc, partition_perc=SUN397_partition_perc, distribute_images=True, distribute_level='level2',
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    datasets_references.append(SUN397_dataset_filtered_partition)
    
    print("\tSUN397 dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(SUN397_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(SUN397_dataset_filtered), SUN397_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(SUN397_dataset_filtered_partition), SUN397_partition_perc))
    print("\t\tclasses partition: {}".format(SUN397_dataset_filtered_partition.classes_count))
    print("\t\tscenes classes hierarchy: {}".format(SUN397_dataset_filtered_partition.classes_hierarchy))
    print("\t\tclasses filtering ratio: {}".format(SUN397_dataset_filtered_partition.filtering_classes_effect))
    
    print("Level0: other")
    print("Level1: Pets")
    Oxford_dataset_original = OxfordIIITPet(root=os.path.join(datasets_root_folder, 'OxfordIII-TPet/data'), 
                    split=["train","test"], split_perc=split_perc)
    Oxford_dataset_filtered = OxfordIIITPet(root=os.path.join(datasets_root_folder, 'OxfordIII-TPet/data'), 
                        split=["train","test"], split_perc=split_perc, 
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    Oxford_partition_perc = min(level0_classes['other']['perc']/3 * total_images / len(Oxford_dataset_filtered), 1.0)
    Oxford_dataset_filtered_partition = OxfordIIITPet(root=os.path.join(datasets_root_folder, 'OxfordIII-TPet/data'), 
                        split=["train","test"], split_perc=split_perc, partition_perc=Oxford_partition_perc, distribute_images=True, distribute_level='level2',
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    datasets_references.append(Oxford_dataset_filtered_partition)

    print("\tOxfordIII-Tpet dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(Oxford_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(Oxford_dataset_filtered), Oxford_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(Oxford_dataset_filtered_partition), Oxford_partition_perc))
    print("\t\tclasses partition: {}".format(Oxford_dataset_filtered_partition.classes_count))
    print("\t\tscenes classes hierarchy: {}".format(Oxford_dataset_filtered_partition.classes_hierarchy))
    print("\t\tclasses filtering ratio: {}".format(Oxford_dataset_filtered_partition.filtering_classes_effect))
    
    print("Level1: Cartoon")
    Cartoon_dataset_original = iCartoonFace(root=os.path.join(datasets_root_folder, 'iCartoonFace/data'), 
                    split=["train","test"], split_perc=split_perc)
    Cartoon_dataset_filtered = iCartoonFace(root=os.path.join(datasets_root_folder, 'iCartoonFace/data'), 
                        split=["train","test"], split_perc=split_perc, 
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    Cartoon_partition_perc = min(level0_classes['other']['perc']/3 * total_images / len(Cartoon_dataset_filtered), 1.0)
    Cartoon_dataset_filtered_partition = iCartoonFace(root=os.path.join(datasets_root_folder, 'iCartoonFace/data'), 
                        split=["train","test"], split_perc=split_perc, partition_perc=Cartoon_partition_perc, distribute_images=True,
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    datasets_references.append(Cartoon_dataset_filtered_partition)

    print("\tiCartoonFace dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(Cartoon_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(Cartoon_dataset_filtered), Cartoon_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(Cartoon_dataset_filtered_partition), Cartoon_partition_perc))
    print("\t\tclasses partition: {}".format(Cartoon_dataset_filtered_partition.classes_count))
    print("\t\tclasses filtering ratio: {}".format(Cartoon_dataset_filtered_partition.filtering_classes_effect))    

    print("Level1: Art")
    ArtImages_dataset_original = ArtImages(root=os.path.join(datasets_root_folder, 'ArtImages/data'), 
                    split=["train","test"], split_perc=split_perc)
    ArtImages_dataset_filtered = ArtImages(root=os.path.join(datasets_root_folder, 'ArtImages/data'), 
                        split=["train","test"], split_perc=split_perc, 
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    ArtImages_partition_perc = min(level0_classes['other']['perc']/3 * total_images / len(ArtImages_dataset_filtered), 1.0)
    ArtImages_dataset_filtered_partition = ArtImages(root=os.path.join(datasets_root_folder, 'ArtImages/data'), 
                        split=["train","test"], split_perc=split_perc, partition_perc=ArtImages_partition_perc, distribute_images=True,
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    datasets_references.append(ArtImages_dataset_filtered_partition)

    print("\tArtImages dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(ArtImages_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(ArtImages_dataset_filtered), ArtImages_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(ArtImages_dataset_filtered_partition), ArtImages_partition_perc))
    print("\t\tclasses partition: {}".format(ArtImages_dataset_filtered_partition.classes_count))
    print("\t\tclasses filtering ratio: {}".format(ArtImages_dataset_filtered_partition.filtering_classes_effect))

   
    # counting overall classes for each level
    final_dataset_level_1_classes_count = {}
    final_dataset_level_2_classes_count = {}
    final_dataset_level_3_classes_count = {}
    for d in datasets_references:
        for t in d.targets:
            if not (d == SIDD_dataset_filtered_partition and (t['level1'] == 'nonselfie' or t['level2'] == 'nonselfie' or t['level3'] == 'nonselfie')): 
                if t['level1'] not in final_dataset_level_1_classes_count:
                    final_dataset_level_1_classes_count[t['level1']] = 1
                elif t['level1'] in final_dataset_level_1_classes_count:
                    final_dataset_level_1_classes_count[t['level1']] += 1

                if t['level2'] not in final_dataset_level_2_classes_count:
                    final_dataset_level_2_classes_count[t['level2']] = 1
                elif t['level2'] in final_dataset_level_2_classes_count:
                    final_dataset_level_2_classes_count[t['level2']] += 1

                if t['level3'] not in final_dataset_level_3_classes_count:
                    final_dataset_level_3_classes_count[t['level3']] = 1
                elif t['level3'] in final_dataset_level_3_classes_count:
                    final_dataset_level_3_classes_count[t['level3']] += 1

    hist_elements = 30
    # print(final_dataset_level_2_classes_count)    
    final_dataset_level_1_classes_count_sorted = {}
    final_dataset_level_1_classes_color = []
    for key, value in sorted(final_dataset_level_1_classes_count.items(), key=lambda item: item[1]):
        final_dataset_level_1_classes_count_sorted[key] = value
        for i, d in enumerate(datasets_references):
            if key in d.classes_map['level1']:
                final_dataset_level_1_classes_color.append(list(dataset_colors.values())[i])

    print("Level 1 Classes Distribution: {}".format(final_dataset_level_1_classes_count_sorted))
    plt.xticks(rotation=45, ha='right')
    plt.bar(list(final_dataset_level_1_classes_count_sorted.keys()), list(final_dataset_level_1_classes_count_sorted.values()), color=final_dataset_level_1_classes_color)
    plt.gca().set(title='Level 1 Partitioned Filtered Classes Distribution', ylabel='# of images')
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(list(final_dataset_level_1_classes_count_sorted.values())):
        plt.text(xlocs[i] - 0.3, v + 0.1, str(v), fontsize=8)
    for i, j in dataset_legend.items(): #Loop over color dictionary
        plt.bar(i, 0,width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("{}/filtered_partition_level1_classes_distribution.svg".format(save_folder))
    plt.close()

    final_dataset_level_2_classes_count_sorted = {}
    final_dataset_level_2_classes_color = []
    for key, value in sorted(final_dataset_level_2_classes_count.items(), key=lambda item: item[1]):
        final_dataset_level_2_classes_count_sorted[key] = value
        for i, d in enumerate(datasets_references):
            if key in d.classes_map['level2']:
                final_dataset_level_2_classes_color.append(list(dataset_colors.values())[i])
    print("Level 2 Classes Distribution: {}".format(final_dataset_level_2_classes_count_sorted))
    plt.xticks(rotation=45, ha='right')
    plt.bar(list(final_dataset_level_2_classes_count_sorted.keys()), list(final_dataset_level_2_classes_count_sorted.values()), color=final_dataset_level_2_classes_color)
    plt.gca().set(title='Level 2 Partitioned Filtered Classes Distribution', ylabel='# of images')
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(list(final_dataset_level_2_classes_count_sorted.values())):
        plt.text(xlocs[i] - 0.3, v + 0.1, str(v), fontsize=4)
    for i, j in dataset_legend.items(): #Loop over color dictionary
        plt.bar(i, 0,width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("{}/filtered_partition_level2_classes_distribution.svg".format(save_folder))
    plt.close()
    
    final_dataset_level_3_classes_count_sorted = {}
    final_dataset_level_3_classes_color = []
    for key, value in sorted(final_dataset_level_3_classes_count.items(), key=lambda item: item[1]):
        final_dataset_level_3_classes_count_sorted[key] = value
        for i, d in enumerate(datasets_references):
            if key in d.classes_map['level3']:
                final_dataset_level_3_classes_color.append(list(dataset_colors.values())[i])
    print("Level 3 Classes Distribution: {}".format(final_dataset_level_3_classes_count_sorted))
    plt.xticks(rotation=45, ha='right')
    plt.bar(list(final_dataset_level_3_classes_count_sorted.keys())[:hist_elements], list(final_dataset_level_3_classes_count_sorted.values())[:hist_elements], color=final_dataset_level_3_classes_color[:hist_elements])
    plt.gca().set(title='Level 3 Partitioned Filtered Classes Distribution', ylabel='# of images')
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(list(final_dataset_level_3_classes_count_sorted.values())[:hist_elements]):
        plt.text(xlocs[i] - 0.3, v + 0.1, str(v), fontsize=4)
    for i, j in dataset_legend.items(): #Loop over color dictionary
        plt.bar(i, 0,width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("{}/filtered_partition_level3_classes_distribution.svg".format(save_folder))
    plt.close()


    # creation of the images csv
    with open(os.path.join(csv_save_path, 'images_metadata.csv'), mode='w') as file:
        csv_writer = csv.writer(file, delimiter=',')
        csv_writer.writerow(['original_dataset', 'img_folder', 'img_name', 'split', 'level0', 'level1', 'level2', 'level3', 'target_level'])

        for m in SIDD_dataset_filtered_partition.metadata:
            if m['target']['level1'] == 'selfie':
                csv_writer.writerow(['Selfie-Image-Detection-Dataset', m['img_folder'], m['img_name'], m['split'], 'people', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level3']])
        for m in EMOTIC_dataset_filtered.metadata:
            csv_writer.writerow(['EMOTIC', m['img_folder'], m['img_name'], m['split'], 'people', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level3']])
        for m in SUN397_dataset_filtered_partition.metadata:
            csv_writer.writerow(['SUN397', m['img_folder'], m['img_name'], m['split'], 'scenes', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level2']])
        for m in Oxford_dataset_filtered_partition.metadata:
            csv_writer.writerow(['OxfordIII-TPet', m['img_folder'], m['img_name'], m['split'], 'other', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level2']])
        for m in Cartoon_dataset_filtered_partition.metadata:
            csv_writer.writerow(['iCartoonFace', m['img_folder'], m['img_name'], m['split'], 'other', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level3']])
        for m in ArtImages_dataset_filtered_partition.metadata:
            csv_writer.writerow(['ArtImages', m['img_folder'], m['img_name'], m['split'], 'other', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level3']])


if __name__ == '__main__':
    main()
