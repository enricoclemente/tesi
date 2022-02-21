import os
from pydoc import classname

import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Union, Tuple
import scipy.io
from PIL import Image

import numpy as np
import csv


"""
    Selfie-Image-Detection-Dataset implementation for pytorch
    site: https://www.kaggle.com/jigrubhatt/selfieimagedetectiondataset
    paper: None

    The dataset is composed of Selfie or Not Selfie images
    This dataset as no hierachical labels

    Statistics: 78619 images
                {'selfie': 46836, 'nonselfie': 31783}
                image with smallest height: Test_data/Selfie/Selfie44658.jpg W=306 H=306
                image with smallest width: Test_data/Selfie/Selfie44658.jpg W=306 H=306
"""
class SIDD(Dataset):
    """
        Selfie-Image-Detection-Dataset Dataset

        Args: 
            root (string): Root directory where images are downloaded to or better to the extracted sun397 folder.
                            folder data structure should be:
                            data (root)
                                |-Test_data
                                |-Training_data
                                |-Validation_data
            split (string or list): possible options: 'train', 'test', if list of multiple splits they will be treated as unique split
            split_perc (float): in order to custom the dataset you can choose the split percentage
            transform (callable, optional): A function/transform that  takes in an PIL image 
                and returns a transformed version. E.g, ``transforms.ToTensor``
    """
    def __init__(self, root: str, split: Union[List[str], str] = "train", split_perc: float = 0.8, transform: Optional[Callable] = None):
        self.root = root

        if isinstance(split, list):
            self.split = split
        else:
            self.split = [split]
        
        assert split_perc <= 1, "Split percentage must be a floating number < 1!"
        self.split_perc = split_perc

        self.transform = transform

        self.metadata, self.targets, self.classes_map, self.classes_count = self._read_metadata()
        self.classes = list(self.classes_map.keys())

    """
        Read all metadata related to dataset in order to compose it
    """
    def _read_metadata(self):
        metadata = []
        targets = []

        # create map of classes { classname: index }
        classes_map = {"selfie": 0, "nonselfie": 1}
        # create map of classes with number of images per classes { classname: # of images in class }
        classes_count = {"selfie": 0, "nonselfie": 0}

        # will be used to distribute in the same proportion each class into ttrain and test
        classes_splitter = {"selfie": 0, "nonselfie": 0}

        for folder_name in ["Test_data", "Training_data", "Validation_data"]:
            for sub_folder_name in ["Selfie", "NonSelfie"]:
                class_name = sub_folder_name.lower()
                for img in os.listdir(os.path.join(self.root, folder_name, sub_folder_name)):
                    classes_count[class_name] += 1
        
        for folder_name in ["Test_data", "Training_data", "Validation_data"]:
            for sub_folder_name in ["Selfie", "NonSelfie"]:
                class_name = sub_folder_name.lower()
                for img in os.listdir(os.path.join(self.root, folder_name, sub_folder_name)):
                    meta = {}
                    if "train" in self.split:
                        if classes_splitter[class_name] < int(classes_count[class_name] * self.split_perc):
                            meta['split'] = "train"
                            meta['img_name'] = img
                            meta['img_folder'] = os.path.join(folder_name, sub_folder_name)
                            # put relative class index in targets since img_folder is equal to class name
                            meta['target'] = {'level1': class_name}
                            targets.append(meta['target'])
                            metadata.append(meta)

                    if "test" in self.split:
                        if classes_splitter[class_name] >= int(classes_count[class_name] * self.split_perc):
                            meta['split'] = "test"
                            meta['img_name'] = img
                            meta['img_folder'] = os.path.join(folder_name, sub_folder_name)
                            # put relative class index in targets since img_folder is equal to class name
                            meta['target'] = {'level1': class_name}
                            targets.append(meta['target'])
                            metadata.append(meta)
                    classes_splitter[class_name] += 1
        # print(classes_count)

        return metadata, targets, classes_map, classes_count
    

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
