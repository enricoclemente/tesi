import os
import collections

import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Union, Tuple
import scipy.io
from PIL import Image

import numpy as np


"""
    OxfordIII-TPet Dataset implementation for pytorch
    site: https://www.robots.ox.ac.uk/~vgg/data/pets/
    paper: file:///Users/enricodeveloper/Documents/Scuola/Tesi/Esperimenti/Datasets%20Papers/OxfordIII-TPet.pdf
    The dataset is focused on fine grained classification of animal breeds.

    Annotations are made for the pet present in the photo. 
    In this implementation I will ignore Trimap and head bbox annotations.

    Statistics:
    
        level1: cat, dog
        level2: {'abyssinian': 198, 'american_bulldog': 200, 'american_pit_bull_terrier': 200, 'basset_hound': 200, 'beagle': 200, 'bengal': 200, 'birman': 200, 'bombay': 184, 'boxer': 199, 'british_shorthair': 200, 'chihuahua': 200, 'egyptian_mau': 190, 'english_cocker_spaniel': 196, 'english_setter': 200, 'german_shorthaired': 200, 'great_pyrenees': 200, 'havanese': 200, 'japanese_chin': 200, 'keeshond': 199, 'leonberger': 200, 'maine_coon': 200, 'miniature_pinscher': 200, 'newfoundland': 196, 'persian': 200, 'pomeranian': 200, 'pug': 200, 'ragdoll': 200, 'russian_blue': 200, 'saint_bernard': 200, 'samoyed': 200, 'scottish_terrier': 199, 'shiba_inu': 200, 'siamese': 199, 'sphynx': 200, 'staffordshire_bull_terrier': 189, 'wheaten_terrier': 200, 'yorkshire_terrier': 200}

"""
class OxfordIIITPet(Dataset):
    """
        OxfordIII-TPet Dataset
        
        Args:
            root (string): Root directory where images are downloaded to or better to the extracted OxfordIII-TPet folder.
                            folder data structure should be:
                                data (root)
                                    |-images
                                    |-annotations

            split (string or list): possible options: 'train', 'test', if list of multiple splits they will be treated as unique split
            split_perc (float): in order to customize the dataset you can choose the split percentage
            transform (callable, optional): A function/transform that  takes in an PIL image 
                and returns a transformed version. E.g, ``transforms.ToTensor``
            partition (float): use it to take only a part of the dataset, keeping the proportion of number of images per classes
                split_perc will work as well splitting the partion
            aspect_ratio_threshold (float): use it to filter images that have a greater aspect ratio
            dim_threshold (float): use it to filter images which area is 
                lower of = dim_threshold * dim_threshold * 1/aspect_ratio_threshold
    """
    def __init__(self, root: str, split: Union[List[str], str] = "train", split_perc: float = 0.8, 
                transform: Optional[Callable] = None, partition_perc: float = 1.0, distribute_images: bool = False, distribute_level: str = 'level3',
                aspect_ratio_threshold: float = None, dim_threshold: int = None):
        self.root = root

        if isinstance(split, list):
            self.split = split
        else:
            self.split = [split]
        
        assert split_perc <= 1, "Split percentage must be a floating number < 1!"
        self.split_perc = split_perc

        self.transform = transform

        self.partition_perc = partition_perc
        self.distribute_images = distribute_images

        self.distribute_level = distribute_level

        if aspect_ratio_threshold is not None:
            self.aspect_ratio_threshold = aspect_ratio_threshold
        else:
            self.aspect_ratio_threshold = None

        if dim_threshold is not None:
            self.area_threshold = dim_threshold * dim_threshold * 1/aspect_ratio_threshold
        else: 
            self.area_threshold = None

        self.metadata, self.targets, self.classes_map, self.classes_count, self.classes_hierarchy, self.filtering_classes_effect, self.total_filtered = self._read_metadata()
        self.classes = list(self.classes_map.keys())

    
    """
        Read all metadata related to dataset in order to compose it
    """
    def _read_metadata(self):
        metadata = []
        targets = []

        total_images = 0
        # create map of classes { classname: index }
        classes_map = {'level1': {'pets': 0}, 'level2': {'cat': 0, 'dog': 0}, 'level3': {}}
        # create map of classes with number of images per classes { classname: # of images in class }
        classes_count = {'level1': {'pets': 0}, 'level2': {'cat': 0, 'dog': 0}, 'level3': {}}

        # will be used to distribute in the same proportion each class into ttrain and test
        classes_splitter = {'level1': {'pets': 0}, 'level2': {'cat': 0, 'dog': 0}, 'level3': {}}

        # structures for tracking filtered images due to thresholds
        filtered_classes_count = {'level1': {'pets': 0}, 'level2': {'cat': 0, 'dog': 0}, 'level3': {}}
        total_filtered = 0

        # in this case there is a two-level hierarchy
        classes_hierarchy = { "cat": [], "dog": []}

        # will be used to distribute equally images among classes
        distributed_classes_count = {'level1': {'pets': 0}, 'level2': {'cat': 0, 'dog': 0}, 'level3': {}}

        # first read annotation files in order to get statistics and info about images
        for file_name in ['trainval.txt', 'test.txt']:
            with open(os.path.join(self.root, "annotations", file_name), 'r') as file:
                for line in file:
                    line = line.strip('\n')
                    line = line.split(' ')
                    img_name = line[0] + '.jpg'
                    class_name = '_'.join(img_name.split('_')[:-1]).lower()
                    class_ID = line[1]
                    specie_ID = line[2]
                    specie_name = list(classes_hierarchy.keys())[int(specie_ID) -1]
                    if class_name not in classes_map['level3'].keys():
                        classes_map['level3'][class_name] = int(class_ID) - 1 # because annotations start from 1 and not 0
                        
                        classes_count['level1']['pets'] += 1
                        classes_count['level2'][specie_name] += 1
                        classes_count['level3'][class_name] = 1

                        classes_splitter['level3'][class_name] = 0

                        filtered_classes_count['level1']['pets'] += 1
                        filtered_classes_count['level2'][specie_name] += 1
                        filtered_classes_count['level3'][class_name] = 1

                        distributed_classes_count['level3'][class_name] = 0

                        if specie_ID == "1":
                            classes_hierarchy["cat"].append(class_name)
                        elif specie_ID == "2":
                            classes_hierarchy["dog"].append(class_name)
                    else:
                        classes_count['level1']['pets'] += 1
                        classes_count['level2'][specie_name] += 1
                        classes_count['level3'][class_name] += 1

                        filtered_classes_count['level1']['pets'] += 1
                        filtered_classes_count['level2'][specie_name] += 1
                        filtered_classes_count['level3'][class_name] += 1
                    
                    total_images += 1

                    W, H = Image.open(os.path.join(self.root, 'images', img_name)).size

                    if self.aspect_ratio_threshold is not None and (W/H > self.aspect_ratio_threshold or W/H < 1/self.aspect_ratio_threshold):
                        filtered_classes_count['level1']['pets'] -= 1
                        filtered_classes_count['level2'][specie_name] -= 1
                        filtered_classes_count['level3'][class_name] -= 1
                        total_filtered += 1
                        total_images -= 1
                    elif self.area_threshold is not None and (W*H < self.area_threshold):
                        filtered_classes_count['level1']['pets'] -= 1
                        filtered_classes_count['level2'][specie_name] -= 1
                        filtered_classes_count['level3'][class_name] -= 1
                        total_filtered += 1
                        total_images -= 1

        total_images = int(total_images * self.partition_perc)

        # try to distributed images equally among classes
        if self.distribute_images == True:
            i = 0
            while i < total_images:
                for c in distributed_classes_count[self.distribute_level].keys():
                    if distributed_classes_count[self.distribute_level][c] < int(filtered_classes_count[self.distribute_level][c]):
                        distributed_classes_count[self.distribute_level][c] += 1
                        i += 1
            filtered_classes_count = distributed_classes_count
        else:
            for c in filtered_classes_count[self.distribute_level].keys():
                filtered_classes_count[self.distribute_level][c] = filtered_classes_count[self.distribute_level][c] * self.partition_perc
        
        # print("classes_map: {} \n classes_count: {} \n classes_hierarchy: {}".format(classes_map, classes_count, classes_hierarchy))
        
        # now creating metadata with images infos
        for file_name in ['trainval.txt', 'test.txt']:
            with open(os.path.join(self.root, "annotations", file_name), 'r') as file:
                for line in file:
                    line = line.strip('\n')
                    line = line.split(' ')
                    img_name = line[0] + ".jpg"

                    W, H = Image.open(os.path.join(self.root, 'images', img_name)).size
                    skip = False
                    if self.aspect_ratio_threshold is not None and (W/H > self.aspect_ratio_threshold or W/H < 1/self.aspect_ratio_threshold):
                        skip = True
                    elif self.area_threshold is not None and (W*H < self.area_threshold):
                        skip = True
                    if skip == False:
                        meta = {}
                        
                        class_name = '_'.join(img_name.split('_')[:-1]).lower()
                        class_ID = line[1]
                        specie_ID = line[2]
                        specie_name = list(classes_hierarchy.keys())[int(specie_ID) -1]
                        if self.distribute_level == 'level1':
                            current_class_name = 'pets'
                        elif self.distribute_level == 'level2':
                            current_class_name = specie_name
                        elif self.distribute_level == 'level3':
                            current_class_name = class_name

                        if "train" in self.split:
                            if classes_splitter[self.distribute_level][current_class_name] < int(filtered_classes_count[self.distribute_level][current_class_name] * self.split_perc):
                                meta['split'] = "train"
                                meta['img_name'] = img_name
                                meta['img_folder'] = 'images'
                                meta['target'] = { 'level1': 'pets',
                                                    'level2': list(classes_hierarchy.keys())[int(specie_ID) -1], 
                                                    'level3': class_name, 
                                                    }
                                
                                targets.append(meta['target'])
                                metadata.append(meta)
                                classes_splitter[self.distribute_level][current_class_name] += 1
                            elif (classes_splitter[self.distribute_level][current_class_name] >= int(filtered_classes_count[self.distribute_level][current_class_name] * self.split_perc) 
                                and classes_splitter[self.distribute_level][current_class_name] < int(filtered_classes_count[self.distribute_level][current_class_name])):
                                if "test" in self.split:
                                        meta['split'] = "test"
                                        meta['img_name'] = img_name
                                        meta['img_folder'] = 'images'
                                        meta['target'] = { 'level1': 'pets',
                                                            'level2': list(classes_hierarchy.keys())[int(specie_ID) -1], 
                                                            'level3':  class_name,
                                                            }
                                        targets.append(meta['target'])
                                        metadata.append(meta)
                                        classes_splitter[self.distribute_level][current_class_name] += 1

        # check how much filtering changed classes proportion
        filtering_classes_effect = {}
        filtering_classes_effect_sorted = collections.OrderedDict()
        for key in classes_count[self.distribute_level].keys():
            if round(filtered_classes_count[self.distribute_level][key]/classes_count[self.distribute_level][key], 2) != 1.0:
                filtering_classes_effect[key] = round(filtered_classes_count[self.distribute_level][key]/classes_count[self.distribute_level][key], 2)

        for key, value in sorted(filtering_classes_effect.items(), key=lambda item: item[1]):
            filtering_classes_effect_sorted[key] = value
        # print(filtered_classes_count)
        # print(filtering_classes_effect_sorted)
        # print(total_filtered)
        # classes_map_hierarchical = {'level1': {'pets': 0}, 'level2': {'cat': 0, 'dog': 0}, 'level3': classes_map}
        return (metadata, targets, classes_map, classes_splitter, classes_hierarchy,
                filtering_classes_effect_sorted, total_filtered)
                

    def __len__(self):
        return len(self.metadata)       
                

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(os.path.join(self.root, self.metadata[idx]['img_folder'], self.metadata[idx]['img_name']))

        if img.mode == "1" or img.mode == "L" or img.mode == "P" or img.mode == "RGBA": # if gray-scale image convert into rgb
            img = img.convert('RGB')

        target = self.targets[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target




