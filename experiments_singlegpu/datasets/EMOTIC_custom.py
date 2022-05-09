import os

import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Union, Tuple
import scipy.io
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt


"""
    EMOTions In Context (EMOTIC) Dataset implementation for pytorch
    site: http://sunai.uoc.edu/emotic/
    paper: http://sunai.uoc.edu/emotic/pdf/emotic_pami2019.pdf
    The dataset is focused on emotions in context
    Warning! Part of dataset is ignored: images in the folder ade20k are not taken 
        because of the purpose of creating SPP dataset. 
        Other dataset SUN397 is composed with ade20k images


    Extre annotations are made for each person in the image and are the following:
    - bounding box: (x,y) starting point, width and height
    - emotion categories: there are 26 different emotion categories
    - continuous dimensions: emotions classified by VAD (Valence, Arousal and Dominance) model, 1 value for each dimension
    - person gender
    - person age

    Statistics:
        23571 images (but removing ade20k images since they are in common with SUN397 there are 17870 images)
        {'nonselfie': 17870}
        The two most smallest images are: 
            with the smallest H: mscoco/images/COCO_val2014_000000118638.jpg W= 640 H= 111 
            with the smallest W: emodb_small/images/6jl8g5davklv3hw4fh.jpg W= 150 H= 150
"""
class EMOTIC(Dataset):
    """
        EMOTions In Context (EMOTIC) Dataset
        
        Args:
            root (string): Root directory where images are downloaded to or better to the extracted cvpr_emotic folder.
                            folder data structure should be:
                                data (root)
                                    |-cvpr_emotic (images)
                                    |   |-ade20k
                                    |   |-emodb_small
                                    |   |-mscoco
                                    |-CVPR17_Annotations.mat (annotations)
            split (string or list): possible options: 'train', 'test', 
                if list of multiple splits they will be treated as unique split
            split_perc (float): in order to custom the dataset you can choose the split percentage
            transform (callable, optional): A function/transform that  takes in an PIL image 
                and returns a transformed version. E.g, ``transforms.ToTensor``
            partition (float): use it to take only a part of the dataset, keeping the proportion of number 
                of images per classes; split_perc will work as well splitting the partion
            distribute_images (bool): decide to distribute images equally among classes (is usefol when taking a partition)
            aspect_ratio_threshold (float): use it to filter images that have a greater aspect ratio
            dim_threshold (float): use it to filter images which area is 
                lower of = dim_threshold * dim_threshold * 1/aspect_ratio_threshold
    """
    split_all = ['train', 'val', 'test']

    emotion_categories_map = {
        "Affection": 0,
        "Anger": 1,
        "Annoyance": 2,
        "Anticipation": 3,
        "Aversion": 4,
        "Confidence": 5,
        "Disapproval": 6,
        "Disconnection": 7,
        "Disquietment": 8,
        "Doubt/Confuzion": 9,
        "Embarrassment": 10,
        "Engagement": 11,
        "Esteem": 12,
        "Excitement": 13,
        "Fatigue": 14,
        "Fear": 15,
        "Happiness": 16,
        "Pain": 17,
        "Peace": 18,
        "Pleasure": 19,
        "Sadness": 20,
        "Sensitivity": 21,
        "Suffering": 22,
        "Surprise": 23,
        "Sympathy": 24, 
        "Yearning": 25,
    }

    def __init__(self, root: str, split: Union[List[str], str] = "train", split_perc: float = 0.8, distribute_images = False,
                transform: Optional[Callable] = None, partition_perc: float = 1.0, aspect_ratio_threshold: float = None, dim_threshold: int = None):
        
        self.root = root

        if isinstance(split, list):
            assert len(split) <=3, "You can specify maximum train, val and test splits together"
            self.split = split
        else:
            self.split = [split]
        
        assert split_perc <= 1, "Split percentage must be a floating number < 1!"
        self.split_perc = split_perc

        self.transform = transform

        self.partition_perc = partition_perc

        if aspect_ratio_threshold is not None:
            self.aspect_ratio_threshold = aspect_ratio_threshold
        else:
            self.aspect_ratio_threshold = None

        if dim_threshold is not None:
            self.area_threshold = dim_threshold * dim_threshold * 1/aspect_ratio_threshold
        else: 
            self.area_threshold = None

        # read annotation file in order to create targets and retrieve images location
        self.metadata, self.targets, self.classes_map, self.classes_count, self.filtering_classes_effect, self.total_filtered = self._read_metadata() 
        self.classes = list(self.classes_count.keys())


    def _read_metadata(self):
        # annotation file is in .mat format. A structured file 
        annotations_file = scipy.io.loadmat(os.path.join(self.root, 'CVPR17_Annotations.mat'))

        metadata = []
        targets_all = []

        total_images = 0
        classes_count = {'nonselfie': 0}

        filtered_classes_count = {"nonselfie": 0}
        total_filtered = 0

        # will be used to distribute in the same proportion each class into train and test
        classes_splitter = {"nonselfie": 0}

        # calculating statistics
        for s in ['train', 'val', 'test']:
            split_annotations = annotations_file[s][0]
            for i in range(len(split_annotations)):

                # skip ade20k images since SUN397 is made also with them
                if split_annotations[i][1][0] != 'ade20k/images':
                    total_images += 1
                    classes_count['nonselfie'] += 1
                    filtered_classes_count['nonselfie'] += 1
                
                    W, H = Image.open(os.path.join(self.root, 'cvpr_emotic', split_annotations[i][1][0], split_annotations[i][0][0])).size
                    
                    if self.aspect_ratio_threshold is not None and (W/H > self.aspect_ratio_threshold or W/H < 1/self.aspect_ratio_threshold):
                        filtered_classes_count['nonselfie'] -= 1
                        total_images -= 1
                        total_filtered += 1
                    elif self.area_threshold is not None and (W*H < self.area_threshold):
                        filtered_classes_count['nonselfie'] -= 1
                        total_images -= 1
                        total_filtered += 1
                
        total_images = int(total_images * self.partition_perc)
        filtered_classes_count['nonselfie'] = total_images

        
        longest_target = 0
        for s in ['train', 'val', 'test']:
            split_annotations = annotations_file[s][0]
            for i in range(len(split_annotations)):

                W, H = Image.open(os.path.join(self.root, 'cvpr_emotic', split_annotations[i][1][0], split_annotations[i][0][0])).size
                skip = False
                if self.aspect_ratio_threshold is not None and (W/H > self.aspect_ratio_threshold or W/H < 1/self.aspect_ratio_threshold):
                    skip = True
                elif self.area_threshold is not None and (W*H < self.area_threshold):
                    skip = True
                
                # skip ade20k images since SUN397 is made also with them
                if split_annotations[i][1][0] == 'ade20k/images':
                    skip = True

                if skip == False:
                    
                    meta = {}
                    take_img = False
                    if "train" in self.split:
                        if classes_splitter['nonselfie'] < int(filtered_classes_count['nonselfie'] * self.split_perc):
                            meta['split'] = 'train'
                            take_img = True
                    if "test" in self.split:
                        if (classes_splitter['nonselfie'] >= int(filtered_classes_count['nonselfie'] * self.split_perc) 
                            and classes_splitter['nonselfie'] < int(filtered_classes_count['nonselfie'])):
                            meta['split'] = 'test'
                            take_img = True
                    if take_img == True:
                        meta['img_name'] = split_annotations[i][0][0]
                        meta['img_folder'] = os.path.join('cvpr_emotic', split_annotations[i][1][0])

                        # in past grayscale images were skipped, now they are used, just converted into RGB
                        # img = Image.open(os.path.join(self.root, 'cvpr_emotic', annotation['img_folder'], annotation['img_name']))
                        # if img.mode == "1" or img.mode == "L" or img.mode == "P": # if gray-scale image skip it
                        #     continue
                        # annotation['img_original_dataset']=target_annotations[i][3][0][0][0][0]   # ignored
                        # list of target because there could be more than one person
                        targets = []
                        for p in range(len(split_annotations[i][4][0])):

                            if (p+1) > longest_target:    # find the maximum number of people between all images
                                longest_target = p+1
                            
                            target = {}
                            
                            target['bbox'] = split_annotations[i][4][0][p][0][0]

                            if s == 'test' or s == 'val': # test annotations are made from several annotators, so the structure is different
                                # combined annotations of emotions categories
                                # print(split_annotations[i][4][0][p][2])
                                # print(split_annotations[i][0][0])
                                # print(split_annotations[i][1][0])
                                if len(split_annotations[i][4][0][p][2]) == 0:  # in test this image /emodb_small/images/12ctuwhai2qxczp2wf.jpg has no emotions annotation
                                    target['emotions'] = []
                                else:
                                    target['emotions'] = split_annotations[i][4][0][p][2][0]
                                # combined annotations of continuous dimensions
                                target['vad'] = split_annotations[i][4][0][p][4][0][0]
                                target['gender'] = split_annotations[i][4][0][p][5][0]
                                target['age'] = split_annotations[i][4][0][p][6][0]
                            else:
                                target['emotions'] = split_annotations[i][4][0][p][1][0][0][0][0]
                                target['vad'] = split_annotations[i][4][0][p][2][0][0]
                                target['gender'] = split_annotations[i][4][0][p][3][0]
                                target['age'] = split_annotations[i][4][0][p][4][0]
                            
                            targets.append(target)
                        meta['target'] = {'level1': 'nonselfie', 
                                            'extra_annotations': targets,
                                            'level2': 'nonselfie', 
                                            'level3': 'nonselfie', }

                        targets_all.append(meta['target'])
                        metadata.append(meta)

                    classes_splitter['nonselfie'] += 1

        # print(longest_target)
        classes_map_hierarchical = {'level1': {"nonselfie": 0}, 'level2': {"nonselfie": 0}, 'level3': {"nonselfie": 0}}
        return metadata, targets_all, classes_map_hierarchical, filtered_classes_count, filtered_classes_count, total_filtered
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(os.path.join(self.root, self.metadata[idx]['img_folder'], self.metadata[idx]['img_name']))
        
        if img.mode == "1" or img.mode == "L" or img.mode == "P": # if gray-scale image convert into RGB
            img = img.convert('RGB')
        
        target = self.targets[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target



class EMOTICPair(EMOTIC):
    """
        Extends EMOTIC Dataset 
        It will return a pair for every image. Is used for self-supervised learning. 
        If you pass at constructor trasform as {"augmentation_1": ..., "agumentation_2": ..} two different trasnformation will be applied.
        If you pass a simple transform, the same transformation will be applied to the two augmentations
    """
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(os.path.join(self.root, 'cvpr_emotic', self.metadata[idx]['img_folder'], self.metadata[idx]['img_name']))

        if self.transform is not None:
            if isinstance(self.transform, dict):
                # if augmentations are different
                augmentation_1 = self.transform['augmentation_1'](img)
                augmentation_2 = self.transform['augmentation_2'](img)
            else:
                augmentation_1 = self.transform(img)
                augmentation_2 = self.transform(img)
        
        return augmentation_1, augmentation_2



def calculate_percentage_person_over_background(root):

    dataset = EMOTIC(root, split=["train", "test"], aspect_ratio_threshold=2.33, dim_threshold=225)

    # print(dataset.metadata[0])
    # print(dataset.metadata[0]['target']['extra_annotations'])
    # extra_annotations = dataset.metadata[0]['target']['extra_annotations']

    # print(len(extra_annotations))
    # print(extra_annotations[0]['bbox'])

    images_people_ratios_total = {}
    images_people_ratios_mscoco = {}
    images_people_ratios_emodb = {}
    for i, (img, target) in enumerate(dataset):
        W, H = img.size
        img_area = W*H
        people_area = 0
        extra_annotations = dataset.metadata[i]['target']['extra_annotations']
        for j in range(len(extra_annotations)):
            people_area += extra_annotations[j]['bbox'][2]*extra_annotations[j]['bbox'][3]
        
        ratio = round(people_area/img_area, 2)
        if ratio not in images_people_ratios_total:
            images_people_ratios_total[ratio] = 0
        else:
            images_people_ratios_total[ratio] += 1
        
        if 'mscoco' in dataset.metadata[i]['img_folder']:
            if ratio not in images_people_ratios_mscoco:
                images_people_ratios_mscoco[ratio] = 0
            else:
                images_people_ratios_mscoco[ratio] += 1
        else:
            if ratio not in images_people_ratios_emodb:
                images_people_ratios_emodb[ratio] = 0
            else:
                images_people_ratios_emodb[ratio] += 1
        # print(area)
        # print(people_area)
        # if i == 0: 
        #     exit()

    images_people_ratios_total_sorted = {}
    for key, value in sorted(images_people_ratios_total.items(), key=lambda item: item[0]):
        images_people_ratios_total_sorted[key] = value
    images_people_ratios_mscoco_sorted = {}
    for key, value in sorted(images_people_ratios_mscoco.items(), key=lambda item: item[0]):
        images_people_ratios_mscoco_sorted[key] = value
    images_people_ratios_emodb_sorted = {}
    for key, value in sorted(images_people_ratios_emodb.items(), key=lambda item: item[0]):
        images_people_ratios_emodb_sorted[key] = value
    # print(images_people_ratios_total_sorted)
    plt.figure('Ratio people/image')
    fig, (ax0, ax1) = plt.subplots(2,1, figsize=(13, 10))
    plot_total, = ax0.plot(images_people_ratios_total_sorted.keys(), images_people_ratios_total_sorted.values(), label='mscoco + emodb', color='green')
    ax0.fill_between(images_people_ratios_total_sorted.keys(), images_people_ratios_total_sorted.values(), color='green')    
    plot_mscoco, = ax1.plot(images_people_ratios_mscoco_sorted.keys(), images_people_ratios_mscoco_sorted.values(), label='mscoco', color='yellow')
    ax1.fill_between(images_people_ratios_mscoco_sorted.keys(), images_people_ratios_mscoco_sorted.values(), color='yellow')
    plot_emodb, = ax1.plot(images_people_ratios_emodb_sorted.keys(), images_people_ratios_emodb_sorted.values(),label='emodb', color='blue')
    ax1.fill_between(images_people_ratios_emodb_sorted.keys(), images_people_ratios_emodb_sorted.values(), color='blue')

    ax0.legend(loc='upper right')
    ax1.legend(loc='upper right')
    ax0.set_xlabel('ratio')
    ax0.set_ylabel('number of images')
    ax1.set_xlabel('ratio')
    ax1.set_ylabel('number of images')
    ax0.set_title('Ratio area occupied by people / image total area')
    plt.savefig("./ratio_people_area_image_area.svg")
    plt.close()
        



        