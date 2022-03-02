from fileinput import filename
import os

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
        EMOTions In Context (EMOTIC) Dataset
        
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
    """
    def __init__(self, root: str, split: Union[List[str], str] = "train", split_perc: float = 0.8, 
                transform: Optional[Callable] = None, partition_perc: float = 1.0):
        self.root = root

        if isinstance(split, list):
            self.split = split
        else:
            self.split = [split]
        
        assert split_perc <= 1, "Split percentage must be a floating number < 1!"
        self.split_perc = split_perc

        self.transform = transform

        self.partition_perc = partition_perc

        self.metadata, self.targets, self.classes_map, self.classes_count, self.classes_hierarchy = self._read_metadata()
        self.classes = list(self.classes_map.keys())

    
    """
        Read all metadata related to dataset in order to compose it
    """
    def _read_metadata(self):
        metadata = []
        targets = []

        # create map of classes { classname: index }
        classes_map = {}
        # create map of classes with number of images per classes { classname: # of images in class }
        classes_count = {}

        # will be used to distribute in the same proportion each class into ttrain and test
        classes_splitter = {}

        # in this case there is a two-level hierarchy
        classes_hierarchy = { "cat": [], "dog": []}

        # first read annotation files in order to get statistics and info about images
        for file_name in ['trainval.txt', 'test.txt']:
            with open(os.path.join(self.root, "annotations", file_name), 'r') as file:
                for line in file:
                    line = line.strip('\n')
                    line = line.split(' ')
                    img_name = line[0]
                    class_name = '_'.join(img_name.split('_')[:-1]).lower()
                    class_ID = line[1]
                    species = line[2]
                    if class_name not in classes_map:
                        classes_map[class_name] = int(class_ID) - 1 # because annotations start from 1 and not 0
                        classes_count[class_name] = 1
                        classes_splitter[class_name] = 0
                        if species == "1":
                            classes_hierarchy["cat"].append(class_name)
                        elif species == "2":
                            classes_hierarchy["dog"].append(class_name)
                    else:
                        classes_count[class_name] += 1
        
        # print("classes_map: {} \n classes_count: {} \n classes_hierarchy: {}".format(classes_map, classes_count, classes_hierarchy))
        
        # now creating metadata with images infos
        for file_name in ['trainval.txt', 'test.txt']:
            with open(os.path.join(self.root, "annotations", file_name), 'r') as file:
                for line in file:
                    meta = {}
                    line = line.strip('\n')
                    line = line.split(' ')
                    img_name = line[0]
                    class_name = '_'.join(img_name.split('_')[:-1]).lower()
                    class_ID = line[1]
                    species = line[2]

                    if "train" in self.split:
                        if classes_splitter[class_name] < int(classes_count[class_name] * self.split_perc * self.partition_perc):
                            meta['split'] = "train"
                            meta['img_name'] = img_name + ".jpg"
                            meta['img_folder'] = ''
                            meta['target'] = { 'level1': list(classes_hierarchy.keys())[int(species) -1], 
                                                'level2': list(classes_map.keys())[int(class_ID) - 1] }
                            
                            targets.append(meta['target'])
                            metadata.append(meta)
                            classes_splitter[class_name] += 1
                        elif (classes_splitter[class_name] >= int(classes_count[class_name] * self.split_perc * self.partition_perc) 
                            and classes_splitter[class_name] < int(classes_count[class_name] * self.partition_perc)):
                            if "test" in self.split:
                                    meta['split'] = "test"
                                    meta['img_name'] = img_name + ".jpg"
                                    meta['img_folder'] = ''
                                    meta['target'] = { 'level1': list(classes_hierarchy.keys())[int(species) -1], 
                                                        'level2': list(classes_map.keys())[int(class_ID) - 1] }
                                    targets.append(meta['target'])
                                    metadata.append(meta)
                                    classes_splitter[class_name] += 1

        return metadata, targets, classes_map, classes_count, classes_hierarchy
                

    def __len__(self):
        return len(self.metadata)       
                

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(os.path.join(self.root, 'images', self.metadata[idx]['img_folder'], self.metadata[idx]['img_name']))

        if img.mode == "1" or img.mode == "L" or img.mode == "P" or img.mode == "RGBA": # if gray-scale image convert into rgb
            img = img.convert('RGB')

        target = self.targets[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target




