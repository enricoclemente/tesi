import os
import shutil
import collections

import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Union, Tuple
import scipy.io
from PIL import Image
import PIL


"""
    Art Images dataset implementation for pytorch
    site: https://cvit.iiit.ac.in/research/projects/cvit-projects/cartoonfaces
    paper: https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2016/Mishra-ECCVW2016.pdf

    The dataset contains art images of 5 different styles

    Labels are not hierarchical: 
        - style (5 classes):
            {'drawings': 0, 'engraving': 1, 'iconography': 2, 'painting': 3, 'sculpture': 4}
    
    Statistics:
        8577 images
        {'drawings': 1229, 'engraving': 841, 'iconography': 2308, 'painting': 2270, 'sculpture': 1929}
        The two most smallest images are: 
            with the smallest H: dataset/dataset_updated/training_set/painting/1884.jpg W= 290 H= 92 
            with the smallest W: dataset/dataset_updated/training_set/sculpture/i - 485.jpeg W= 37 H= 320
        The classes have been splitted in the following numbers: 
            {'drawings': {'train': 983, 'test': 246}, 'engraving': {'train': 672, 'test': 169}, 'iconography': {'train': 1846, 'test': 462}, 'painting': {'train': 1816, 'test': 454}, 'sculpture': {'train': 1543, 'test': 386}}

    Be careful some images are corrupted, I moved them in a separate folder
"""
class ArtImages(Dataset):
    """
        Art Images Dataset

        Args: 
            root (string): Root directory where images are downloaded to or better to the extracted sun397 folder.
                            folder data structure should be:
                            data (root)
                                |-dataset
                                |-...
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
        classes_map = {'drawings': 0, 'engraving': 1, 'iconography': 2, 'painting': 3, 'sculpture': 4}
        # create map of classes with number of images per classes { classname: # of images in class }
        classes_count = {'drawings': 0, 'engraving': 0, 'iconography': 0, 'painting': 0, 'sculpture': 0}

        # will be used to distribute in the same proportion each class into train and test
        classes_splitter = {'drawings': 0, 'engraving': 0, 'iconography': 0, 'painting': 0, 'sculpture': 0}

        # structures for tracking filtered images due to thresholds
        filtered_classes_count = {'drawings': 0, 'engraving': 0, 'iconography': 0, 'painting': 0, 'sculpture': 0}
        total_filtered = 0

        # will be used to distribute equally images among classes
        distributed_classes_count = {'drawings': 0, 'engraving': 0, 'iconography': 0, 'painting': 0, 'sculpture': 0}

        for folder_name in ['training_set', 'validation_set']:
            for class_name in classes_map.keys():
                for img in os.listdir(os.path.join(self.root, 'dataset', 'dataset_updated', folder_name, class_name)):
                    classes_count[class_name] += 1
                    filtered_classes_count[class_name] += 1

                    total_images += 1
                    W, H = Image.open(os.path.join(self.root, 'dataset', 'dataset_updated', folder_name, class_name, img)).size

                    if self.aspect_ratio_threshold is not None and (W/H > self.aspect_ratio_threshold or W/H < 1/self.aspect_ratio_threshold):
                        filtered_classes_count[class_name] -= 1
                        total_filtered += 1
                        total_images -= 1
                    elif self.area_threshold is not None and (W*H < self.area_threshold):
                        filtered_classes_count[class_name] -= 1
                        total_filtered += 1
                        total_images -= 1
        
        total_images = int(total_images * self.partition_perc)

        # try to distributed images equally among classes
        if self.distribute_images == True:
            i = 0
            while i < total_images:
                for c in distributed_classes_count.keys():
                    if distributed_classes_count[c] < filtered_classes_count[c]:
                        distributed_classes_count[c] += 1
                        i += 1
            filtered_classes_count = distributed_classes_count
        else:
            for c in filtered_classes_count.keys():
                filtered_classes_count[c] = filtered_classes_count[c] * self.partition_perc
        
        for folder_name in ['training_set', 'validation_set']:
            for class_name in classes_map.keys():
                for img in os.listdir(os.path.join(self.root, 'dataset', 'dataset_updated', folder_name, class_name)):
                    W, H = Image.open(os.path.join(self.root, 'dataset', 'dataset_updated', folder_name, class_name, img)).size
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
                                meta['img_folder'] = os.path.join('dataset', 'dataset_updated', folder_name, class_name)
                                # put relative class index in targets since img_folder is equal to class name
                                meta['target'] = {'level1': 'art', 'level2': class_name,'level3': class_name}
                                targets.append(meta['target'])
                                metadata.append(meta)
                                classes_splitter[class_name] += 1
                            elif (classes_splitter[class_name] >= int(filtered_classes_count[class_name] * self.split_perc) 
                                and classes_splitter[class_name] < int(filtered_classes_count[class_name])):
                                if "test" in self.split:
                                    meta['split'] = "test"
                                    meta['img_name'] = img
                                    meta['img_folder'] = os.path.join('dataset', 'dataset_updated', folder_name, class_name)
                                    # put relative class index in targets since img_folder is equal to class name
                                    meta['target'] = {'level1': 'art', 'level2': class_name,'level3': class_name}
                                    targets.append(meta['target'])
                                    metadata.append(meta)
                                    classes_splitter[class_name] += 1
        
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

        classes_map_hierarchical = {'level1': {'art': 0}, 'level2': classes_map, 'level3': classes_map}
        return (metadata, targets, classes_map_hierarchical, classes_splitter, 
            filtering_classes_effect_sorted, total_filtered)


    def __len__(self):
        return len(self.metadata)       
                

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = None
        try:
            img = Image.open(os.path.join(self.root, self.metadata[idx]['img_folder'], self.metadata[idx]['img_name']))
            # if img.mode == "1" or img.mode == "L" or img.mode == "P" or img.mode == "RGBA" or img.mode == "I" or img.mode == "F": 
            #     # if gray-scale image convert into rgb
            #     img = img.convert('RGB')
            img = img.convert('RGB')
        except PIL.UnidentifiedImageError:
            os.rename(os.path.join(self.root, self.metadata[idx]['img_folder'], self.metadata[idx]['img_name']), os.path.join(self.root, 'corrupted', self.metadata[idx]['img_name']))
            print(os.path.join(self.root, self.metadata[idx]['img_folder'], self.metadata[idx]['img_name']))

        target = self.targets[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
    
