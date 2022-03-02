import os

import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Union, Tuple
import scipy.io
from PIL import Image

import numpy as np

"""
    EMOTions In Context (EMOTIC) Dataset implementation for pytorch
    site: http://sunai.uoc.edu/emotic/
    paper: http://sunai.uoc.edu/emotic/pdf/emotic_pami2019.pdf
    The dataset is focused on emotions in context
    The dataset is split in train, val, and test (respectively 70%, 10%, 20%)

    Annotations are made for each person in the image and are the following:
    - bounding box: (x,y) starting point, width and height
    - emotion categories: there are 26 different emotion categories
    - continuous dimensions: emotions classified by VAD (Valence, Arousal and Dominance) model, 1 value for each dimension
    - person gender
    - person age

    Statistics:
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
            split (string or list): possible options: 'train', 'val', 'test', if list of multiple splits they will be treated as unique split
            target_type (string or list, optional): Type of target to use, ``bbox``, ``emotions``, ``vad``,
                ``gender``, ``age``. Can also be a list to output a tuple with all specified target types.
                The targets represent:
                    - ``bbox`` (np.array shape=(3,) dtype=float32): x, y, width, height of the bbox
                    - ``emotions`` (np.array shape=(26,) dtype=int): emotion categories binary (0, 1) labels
                    - ``vad`` (np.array shape=(3,) dtype=int): vad values
                    - ``gender`` (np.array shape=(2,) dtype=int): person gender binary (0, 1) labels mutually exclusives
                    - ``age`` (np.array shape=(3,) dtype=int): person age range binary (0, 1) labels mutually exclusives
                Target will be an array of targets based on the number of people in the image
            transform (callable, optional): A function/transform that  takes in an PIL image 
                and returns a transformed version. E.g, ``transforms.ToTensor``
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

    def __init__(self, root: str, split: Union[List[str], str] = "train", target_type: Union[List[str], str] = None, transform: Optional[Callable] = None):
        
        self.root = root

        if isinstance(split, list):
            assert len(split) <=3, "You can specify maximum train, val and test splits together"
            self.split = split
        else:
            self.split = [split]
        
        self.target_type = target_type
        self.transform = transform

        # read annotation file in order to create targets and retrieve images location
        self.metadata, self.targets, self.longest_target = self._read_metadata() 
        self.classes = list(self.emotion_categories_map.keys())


    def _read_metadata(self):
        # annotation file is in .mat format. A structured file 
        annotations_file = scipy.io.loadmat(os.path.join(self.root, 'CVPR17_Annotations.mat'))

        annotations = list(dict())
        targets_all = []

        longest_target = 0
        for s in self.split:
            split_annotations = annotations_file[s][0]
            for i in range(len(split_annotations)):
                annotation = {}
                annotation['split'] = s
                annotation['img_name'] = split_annotations[i][0][0]
                annotation['img_folder'] = split_annotations[i][1][0]

                # in past grayscale images were skipped, now they are used, just converted into RGB
                # img = Image.open(os.path.join(self.root, 'cvpr_emotic', annotation['img_folder'], annotation['img_name']))
                # if img.mode == "1" or img.mode == "L" or img.mode == "P": # if gray-scale image skip it
                #     continue

                annotation['img_width'] = split_annotations[i][2][0][0][1][0][0]
                annotation['img_height'] = split_annotations[i][2][0][0][0][0][0]
                # annotation['img_original_dataset']=target_annotations[i][3][0][0][0][0]   # ignored
                # list of target because there could be more than one person
                targets = []
                skip = False
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
                            skip = True
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
                annotation['target'] = {'level1': None, 
                                        'level1_attributes': targets}
                if skip == False:
                    targets_all.append(annotation['target'])
                    annotations.append(annotation)

        # print(longest_target)
        return annotations, targets_all, longest_target
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(os.path.join(self.root, 'cvpr_emotic', self.metadata[idx]['img_folder'], self.metadata[idx]['img_name']))
        
        if img.mode == "1" or img.mode == "L" or img.mode == "P": # if gray-scale image convert into RGB
            img = img.convert('RGB')
        
        target = self.targets[idx]
        # padding target to have all targets of the same dimension
        # if len(target) < self.longest_target:
        #     for i in range(self.longest_target - len(target)):
        #         target.append(-1)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target



class EMOTICPair(EMOTIC):
    """
        Extends EMOTIC Dataset 
        Is used for unsupervised learning. It will return two augmentation of the i-th image. 
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


        



        