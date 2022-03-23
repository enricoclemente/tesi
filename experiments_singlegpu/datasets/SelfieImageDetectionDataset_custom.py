import os
import collections

import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Union, Tuple
import scipy.io
from PIL import Image


"""
    Selfie-Image-Detection-Dataset implementation for pytorch
    site: https://www.kaggle.com/jigrubhatt/selfieimagedetectiondataset
    paper: None

    The dataset is composed of Selfie or Not Selfie images
    This dataset as no hierachical labels

    Labels are simply two:
        - selfie or nonselfie 

    Statistics: 
        78619 images
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
            split (string or list): possible options: 'train', 'test', 
                if list of multiple splits they will be treated as unique split
            split_perc (float): in order to custom the dataset you can choose the split percentage
            transform (callable, optional): A function/transform that  takes in an PIL image 
                and returns a transformed version. E.g, ``transforms.ToTensor``
            partition (float): use it to take only a part of the dataset, keeping the proportion of number of images per classes
                split_perc will work as well splitting the partion
            aspect_ratio_threshold (float): use it to filter images that have a greater aspect ratio
            dim_threshold (float): use it to filter images which area is 
                lower of = dim_threshold * dim_threshold * 1/aspect_ratio_threshold
    """
    def __init__(self, root: str, split: Union[List[str], str] = "train", split_perc: float = 0.8, 
                transform: Optional[Callable] = None, partition_perc: float = 1.0, distribute_images: bool = False,
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

        if aspect_ratio_threshold is not None:
            self.aspect_ratio_threshold = aspect_ratio_threshold
        else:
            self.aspect_ratio_threshold = None

        if dim_threshold is not None:
            self.area_threshold = dim_threshold * dim_threshold * 1/aspect_ratio_threshold
        else: 
            self.area_threshold = None

        self.metadata, self.targets, self.classes_map, self.classes_count, self.filtering_classes_effect, self.total_filtered = self._read_metadata()
        self.classes = list(self.classes_map.keys())


    """
        Read all metadata related to dataset in order to compose it
    """
    def _read_metadata(self):
        metadata = []
        targets = []

        total_images = 0
        # create map of classes { classname: index }
        classes_map = {"selfie": 0, "nonselfie": 1}
        # create map of classes with number of images per classes { classname: # of images in class }
        classes_count = {"selfie": 0, "nonselfie": 0}
        filtered_classes_count = {"selfie": 0, "nonselfie": 0}
        total_filtered = 0

        # will be used to distribute in the same proportion each class into train and test
        classes_splitter = {"selfie": 0, "nonselfie": 0}

        # will be used to distribute equally images among classes
        distributed_classes_count = {"selfie": 0, "nonselfie": 0}

        for folder_name in ["Test_data", "Training_data", "Validation_data"]:
            for sub_folder_name in ["Selfie", "NonSelfie"]:
                class_name = sub_folder_name.lower()
                for img in os.listdir(os.path.join(self.root, folder_name, sub_folder_name)):
                    total_images += 1
                    classes_count[class_name] += 1
                    filtered_classes_count[class_name] += 1

                    W, H = Image.open(os.path.join(self.root, folder_name, sub_folder_name, img)).size

                    if self.aspect_ratio_threshold is not None and (W/H > self.aspect_ratio_threshold or W/H < 1/self.aspect_ratio_threshold):
                        filtered_classes_count[class_name] -= 1
                        total_images -= 1
                        total_filtered += 1
                    elif self.area_threshold is not None and (W*H < self.area_threshold):
                        filtered_classes_count[class_name] -= 1
                        total_images -= 1
                        total_filtered += 1
        
        total_images = int(total_images * self.partition_perc)
        # print(total_images)
        # print(filtered_classes_count)
        # print(total_filtered)
        
        # try to distributed images equally among classes
        if self.distribute_images == True:
            i = 0
            while i < total_images:
                for c in distributed_classes_count.keys():
                    if distributed_classes_count[c] < int(filtered_classes_count[c]):
                        distributed_classes_count[c] += 1
                        i += 1
            filtered_classes_count = distributed_classes_count
        else:
            for c in filtered_classes_count.keys():
                filtered_classes_count[c] = filtered_classes_count[c] * self.partition_perc
        
        for folder_name in ["Test_data", "Training_data", "Validation_data"]:
            for sub_folder_name in ["Selfie", "NonSelfie"]:
                class_name = sub_folder_name.lower()
                for img in os.listdir(os.path.join(self.root, folder_name, sub_folder_name)): 
                    W, H = Image.open(os.path.join(self.root, folder_name, sub_folder_name, img)).size
                    skip = False
                    if self.aspect_ratio_threshold is not None and (W/H > self.aspect_ratio_threshold or W/H < 1/self.aspect_ratio_threshold):
                        skip = True
                    elif self.area_threshold is not None and (W*H < self.area_threshold):
                        skip = True
                    if skip == False:
                        meta = {}
                        if "train" in self.split:
                            if classes_splitter[class_name] < int(filtered_classes_count[class_name] * self.split_perc):
                                meta['split'] = "train"
                                meta['img_name'] = img
                                meta['img_folder'] = os.path.join(folder_name, sub_folder_name)
                                # put relative class index in targets since img_folder is equal to class name
                                meta['target'] = {'level1': class_name, 'level2': class_name, 'level3': class_name,}
                                targets.append(meta['target'])
                                metadata.append(meta)
                                classes_splitter[class_name] += 1
                            elif (classes_splitter[class_name] >= int(filtered_classes_count[class_name] * self.split_perc) 
                                and classes_splitter[class_name] < int(filtered_classes_count[class_name])):
                                if "test" in self.split:
                                    meta['split'] = "test"
                                    meta['img_name'] = img
                                    meta['img_folder'] = os.path.join(folder_name, sub_folder_name)
                                    # put relative class index in targets since img_folder is equal to class name
                                    meta['target'] = {'level1': class_name, 'level2': class_name, 'level3': class_name,}
                                    targets.append(meta['target'])
                                    metadata.append(meta)
                                    classes_splitter[class_name] += 1
                    
        # print(classes_count)
        # check how much filtering changed classes proportion
        filtering_classes_effect = {}
        filtering_classes_effect_sorted = collections.OrderedDict()
        for key in classes_count.keys():
            if round(filtered_classes_count[key]/classes_count[key], 2) != 1.0:
                filtering_classes_effect[key] = round(filtered_classes_count[key]/classes_count[key], 2)

        for key, value in sorted(filtering_classes_effect.items(), key=lambda item: item[1]):
            filtering_classes_effect_sorted[key] = value
        
        # print(filtered_classes_count)
        # print(filtering_classes_effect_sorted)
        # print(total_filtered)
        classes_map_hierarchical = {'level1': classes_map, 'level2': classes_map, 'level3': classes_map}
        return (metadata, targets, classes_map_hierarchical, classes_splitter,
            filtering_classes_effect_sorted, total_filtered)
    

    def __len__(self):
        return len(self.metadata)       
                

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(os.path.join(self.root, self.metadata[idx]['img_folder'], self.metadata[idx]['img_name']))

        if img.mode == "1" or img.mode == "L" or img.mode == "P" or img.mode == "RGBA": 
            # if gray-scale image convert into rgb
            img = img.convert('RGB')

        target = self.targets[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
