import os
import csv
import random

import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Union
from PIL import Image


# setting random seed in order to have always the same random order
random.seed(22)

"""
    Social Profile Pictures Dataset

    The dataset is a composite one, it is created from several datasets in order to reproduce a
    real life scenario of social media profile pictures like Facebook Profile Pictures

    Labels:
        There are hierchical labels on 3 levels, target labels are chosen before during the creation 
        of metadata of the dataset 
"""
class SocialProfilePictures(Dataset):
    """
    Social Profile Pictures Dataset

    Args: 
        root (string): Root directory where various dataset are downloaded
                        Folders data structure should be:
                        datasets (root)
                            |-ArtImages
                            |-IIIT-CFW
                            |-OxofrdIII-TPet
                            |-Selfie-Image-Detection-Dataset
                            |-SUN397
                            |-SocialProfilePictures ()
        split (string or list): possible options: 'train', 'val', 'test'
            if list of multiple splits they will be treated as unique split
        split_perc (float): in order to custom the dataset you can choose the split percentage
        transform (callable, optional): A function/transform that  takes in an PIL image 
            and returns a transformed version. E.g, ``transforms.ToTensor``
        partition_perc (float): use it to take only a part of the dataset, keeping the proportion of number of images per classes
            split_perc will work as well splitting the partion 
        aspect_ratio_threshold (float): use it to filter images that have a greater aspect ratio
        dim_threshold (float): use it to filter images which area is 
            lower of = dim_threshold * dim_threshold * 1/aspect_ratio_threshold
        version (int): indicates which version of the dataset to use 
        shuffle_imgs (bool): if true shuffle images per class in order to have subclasses randomization in different dataset splits
    Attributes:
        - metadata
        - targets
        - classes_map
        - classes_count
        - classes
    """
    def __init__(self, root: str, split: Union[List[str], str] = "train", split_perc: float = 0.8,
                transform: Optional[Callable] = None, partition_perc: float = 1.0,
                aspect_ratio_threshold: float = None, dim_threshold: int = None, version: int = None, randomize_metadata: bool = None):
        
        if version == None:
            raise NotImplementedError("You must choose a dataset version!")
        
        if randomize_metadata == None:
            raise NotImplementedError("You must choose if randomize data")

        self.root = root

        if isinstance(split, list):
            self.split = split
        else:
            self.split = [split]
        
        assert split_perc <= 1, "Split percentage must be a floating number < 1!"
        self.split_perc = split_perc

        self.transform = transform

        self.partition_perc = partition_perc #??TODO the partition in this dataset is not implemented

        if aspect_ratio_threshold is not None:
            self.aspect_ratio_threshold = aspect_ratio_threshold
        else:
            self.aspect_ratio_threshold = None

        if dim_threshold is not None:
            self.area_threshold = dim_threshold * dim_threshold * 1/aspect_ratio_threshold
        else: 
            self.area_threshold = None
        
        self.version = "version_0"+str(version) if version < 10 else "version_"+str(version)
        self.randomize_metadata = randomize_metadata

        self.metadata, self.targets, self.classes_map, self.classes_count = self._read_metadata()
        self.classes = list(self.classes_map.keys())
        
    
    """
        Read all metadata related to dataset in order to compose it
    """
    def _read_metadata(self):
        metadata = []
        targets = []

        # mapping only target levels, other level info will be added in metadata
        classes_map = {}
        classes_metadata = {}
        classes_count = {}
        classes_split_counter = {}

        classes_counter = {}

        # calculating statistics
        with open(os.path.join(self.root, 'SocialProfilePictures', 'data', self.version, 'images_metadata.csv')) as file:
            csv_reader = csv.reader(file, delimiter=',')
            line_count = 0

            class_index = 0
            
            for row in csv_reader:
                if line_count == 0:
                    # ignore header
                    line_count += 1
                else:
                    meta = {}
                    if row[8] not in classes_map:
                        classes_map[row[8]] = class_index
                        classes_metadata[row[8]] = []
                        class_index += 1
                        classes_count[row[8]] = 1
                        classes_counter[row[8]] = 0
                        classes_split_counter[row[8]] = 0
                    else: 
                        classes_count[row[8]] += 1
                        
                    meta['img_name'] = row[2]
                    meta['img_folder'] = os.path.join(row[0], 'data', row[1])
                    meta['target'] = {'level0': row[4], 'level1': row[5], 'level2': row[6], 'level3': row[7], 'target_level': row[8]}
                    classes_metadata[row[8]].append(meta)
                    line_count += 1
            #??print(line_count)
            
        # print(classes_map)
        # print(classes_count)
        #??print(classes_splitter)

        if self.randomize_metadata:
            for class_name in classes_metadata.keys():
                random.shuffle(classes_metadata[class_name])

        for class_name in classes_metadata.keys():
            for meta in classes_metadata[class_name]:
                take_img = False
                if "train" in self.split:
                    if classes_counter[class_name] < int(classes_count[class_name] * self.split_perc):
                        meta['split'] = 'train'
                        take_img = True
                if "val" in self.split:
                    if (classes_counter[class_name] >= int(classes_count[class_name] * self.split_perc) and 
                        classes_counter[class_name] < int(classes_count[class_name] * self.split_perc + classes_count[class_name] * (1.0 - self.split_perc)/2)):
                        meta['split'] = 'val'
                        take_img = True
                if "test" in self.split:
                    if (classes_counter[class_name] >= int(classes_count[class_name] * self.split_perc + classes_count[class_name] * (1.0 - self.split_perc)/2)):
                        meta['split'] = 'test'
                        take_img = True
                if take_img == True:
                    targets.append(classes_map[class_name])
                    metadata.append(meta)
                    classes_split_counter[class_name] += 1
            
                classes_counter[class_name] += 1

        return metadata, targets, classes_map, classes_split_counter


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


class SocialProfilePicturesPro(SocialProfilePictures):
    """
        Extends SocialProfilePictures Dataset 
        It will return image, target and indices of the images. 
        This is useful to investigate when using data loader with suffle and order of metadata is lost
    """
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(os.path.join(self.root, self.metadata[idx]['img_folder'], self.metadata[idx]['img_name']))

        if img.mode == "1" or img.mode == "L" or img.mode == "P" or img.mode == "RGBA": # if gray-scale image convert into rgb
            img = img.convert('RGB')

        target = self.targets[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target, idx


class SocialProfilePicturesPair(SocialProfilePictures):
    """
        Extends SocialProfilePictures Dataset 
        It will return a pair for every image. Is used for self-supervised learning. 
        If you pass at constructor trasform as {"augmentation_1": ..., "agumentation_2": ..} two different transformation will be applied.
        If you pass a simple transform, the same transformation will be applied to the pair
    """
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(os.path.join(self.root, self.metadata[idx]['img_folder'], self.metadata[idx]['img_name']))

        if img.mode == "1" or img.mode == "L" or img.mode == "P" or img.mode == "RGBA": # if gray-scale image convert into rgb
            img = img.convert('RGB')

        if self.transform is not None:
            if isinstance(self.transform, dict):
                # if augmentations are different
                augmentation_1 = self.transform['augmentation_1'](img)
                augmentation_2 = self.transform['augmentation_2'](img)
            else:
                augmentation_1 = self.transform(img)
                augmentation_2 = self.transform(img)
        
        return augmentation_1, augmentation_2

