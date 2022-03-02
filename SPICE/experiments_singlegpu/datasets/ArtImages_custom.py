import os
import shutil

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
    """
    def __init__(self, root: str, split: Union[List[str], str] = "train", 
        split_perc: float = 0.8, transform: Optional[Callable] = None):

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
        classes_map = {'drawings': 0, 'engraving': 1, 'iconography': 2, 'painting': 3, 'sculpture': 4}
        # create map of classes with number of images per classes { classname: # of images in class }
        classes_count = {'drawings': 0, 'engraving': 0, 'iconography': 0, 'painting': 0, 'sculpture': 0}

        # will be used to distribute in the same proportion each class into train and test
        classes_splitter = {'drawings': 0, 'engraving': 0, 'iconography': 0, 'painting': 0, 'sculpture': 0}

        for folder_name in ['training_set', 'validation_set']:
            for class_name in classes_map.keys():
                for img in os.listdir(os.path.join(self.root, 'dataset', 'dataset_updated', folder_name, class_name)):
                    classes_count[class_name] += 1
        
        for folder_name in ['training_set', 'validation_set']:
            for class_name in classes_map.keys():
                for img in os.listdir(os.path.join(self.root, 'dataset', 'dataset_updated', folder_name, class_name)):
                    meta = {}
                    if "train" in self.split:
                        if classes_splitter[class_name] < int(classes_count[class_name] * self.split_perc):
                            meta['split'] = "train"
                            meta['img_name'] = img
                            meta['img_folder'] = os.path.join('dataset', 'dataset_updated', folder_name, class_name)
                            # put relative class index in targets since img_folder is equal to class name
                            meta['target'] = {'level1': class_name}
                            targets.append(meta['target'])
                            metadata.append(meta)

                    if "test" in self.split:
                        if classes_splitter[class_name] >= int(classes_count[class_name] * self.split_perc):
                            meta['split'] = "test"
                            meta['img_name'] = img
                            meta['img_folder'] = os.path.join('dataset', 'dataset_updated', folder_name, class_name)
                            # put relative class index in targets since img_folder is equal to class name
                            meta['target'] = {'level1': class_name}
                            targets.append(meta['target'])
                            metadata.append(meta)
                    classes_splitter[class_name] += 1

        return metadata, targets, classes_map, classes_count

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
    
