import os

import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Union, Tuple
import scipy.io
from PIL import Image

import numpy as np
import csv


"""
    Scenes UNderstanding of 397 Scenes (SUN397) Dataset implementation for pytorch
    site: https://vision.princeton.edu/projects/2010/SUN/
    paper: https://vision.princeton.edu/projects/2010/SUN/paperIJCV.pdf
    The dataset has 397 scenes. Images are not equally distributed in scenes since
    some scenes are more frequent in every-day life then others

    Annotations are made for each scene, but there is also available a three-level hierarchy 
    annotation: https://vision.princeton.edu/projects/2010/SUN/hierarchy/

"""
class SUN397(Dataset):
    """
        Scenes UNderstanding of 397 Scenes (SUN397) Dataset

        Args: 
            root (string): Root directory where images are downloaded to or better to the extracted sun397 folder.
                            folder data structure should be:
                            data (root)
                                |-sun397
                                |   |-partitions (annotations)
                                |   |-SUN397 (images)
            split (string or list): possible options: 'train', 'test', if list of multiple splits they will be treated as unique split
            split_perc (float): since there is no file of dividing train and test images (there are partitions but they are for other purpose), 
                you choose percentage
            transform (callable, optional): A function/transform that  takes in an PIL image 
                and returns a transformed version. E.g, ``transforms.ToTensor``
            target_type (string or list, optional): Type of target to use, ``level1``, ``level2``, ``level3``

    """

    def __init__(self, root: str, split: Union[List[str], str] = "train", split_perc: float = 0.8, target_type: Union[List[str], str] = None, transform: Optional[Callable] = None):
        self.root = root

        if isinstance(split, list):
            self.split = split
        else:
            self.split = [split]

        assert split_perc <= 1, "Split percentage must be a floating number < 1!"
        self.split_perc = split_perc

        self.target_type = target_type
        self.transform = transform

        self.metadata, self.targets, self.classes_map, self.classes_count = self._read_metadata()
        self.classes = list(self.classes_map.keys())


    """
        Read all metadata related to dataset:
        - classes
        - hierarchical classes
    """
    def _read_metadata(self):
        metadata = []
        targets_all = []

        # create map of classes { classname: index }
        classes_map = {}
        # create map of classes with number of images per classes { classname: # of images in class }
        classes_count = {}
        with open(os.path.join(self.root,"sun397/partitions/ClassName.txt"), 'r') as file:
            class_index = 0
            for line in file:
                line = line.strip('\n')
                classes_map[line] = class_index
                classes_count[line] = 0
                class_index += 1
        
        
        for c in classes_map.keys():
            for img in os.listdir(os.path.join(self.root, 'sun397/SUN397', c[1:])):
                classes_count[c] += 1
            
            split_counter = 0
            for img in os.listdir(os.path.join(self.root, 'sun397/SUN397', c[1:])):
                meta = {}
                if "train" in self.split:
                    if split_counter < int(classes_count[c] * self.split_perc):
                        meta['split'] = "train"
                        meta['img_name'] = img
                        meta['img_folder'] = c
                        # put relative class index in targets since img_folder is equal to class name
                        meta['target'] = classes_map[meta['img_folder']]
                        targets_all.append(meta['target'])
                        metadata.append(meta)

                if "test" in self.split:
                    if split_counter >= int(classes_count[c] * self.split_perc):
                        meta['split'] = "test"
                        meta['img_name'] = img
                        meta['img_folder'] = c
                        # put relative class index in targets since img_folder is equal to class name
                        meta['target'] = classes_map[meta['img_folder']]
                        targets_all.append(meta['target'])
                        metadata.append(meta)
                split_counter += 1

        return metadata, targets_all, classes_map, classes_count
                

    def __len__(self):
        return len(self.metadata)
                
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(os.path.join(self.root, 'sun397', 'SUN397', self.metadata[idx]['img_folder'][1:], self.metadata[idx]['img_name']))

        if img.mode == "1" or img.mode == "L" or img.mode == "P" or img.mode == "RGBA": # if gray-scale image convert into rgb
            img = img.convert('RGB')

        target = self.targets[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target






    