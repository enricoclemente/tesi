import os
import shutil


import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Union, Tuple
import scipy.io
from PIL import Image
import PIL

from experiments_singlegpu.datasets.SelfieImageDetectionDataset_custom import SIDD
from experiments_singlegpu.datasets.SelfieImageDetectionDataset_custom import SIDD_total_images

from experiments_singlegpu.datasets.SUN397_custom import SUN397
from experiments_singlegpu.datasets.SUN397_custom import SUN397_total_images

# indicates level0 classes and relative percentual presence in the final dataset
level0_classes = {  'people': { 'perc': 0.6 },
                    'scenes': { 'perc': 0.2 },
                    'other': { 'perc': 0.2 }}

"""
    Create Images CSV of the final dataset starting from the datasets
    Args:
        - total_images: the number of total images of the final dataset
        - split_perc: the percetange of split between train images and test images
    
    The images in the classes will keep the same proportions as from the original statistics
"""
def create_images_csv(total_images=60000, split_perc=0.8):
    # First open the datasets and gets overall statistics
    
    # Working on level0: people
    print("Level0: people")
    # Take a partition of SIDD dataset == total_images * people perc
    SIDD_partition_perc = level0_classes['people']['perc'] * total_images / SIDD_total_images
    SIDD_dataset = SIDD(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/Selfie-Image-Detection-Dataset/data',
                    split=["train", "test"], split_perc=split_perc, partition_perc=SIDD_partition_perc)
    print("\tSIDD dataset statistics")
    print("\tSIDD total_images: {}".format(SIDD_total_images))
    print("\tpartition images: {}".format(len(SIDD_dataset)))
    print("\tclasses partition: {}".format(SIDD_dataset.classes_count))

    # Working on level0: scenes
    print("Level0: scenes")
    SUN397_partition_perc = level0_classes['scenes']['perc'] * total_images / SUN397_total_images
    SUN397_dataset = SUN397(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/SUN397/data', 
                        split=["train","test"], split_perc=split_perc, partition_perc=SUN397_partition_perc)
    print("\tSUN397 dataset statistics")
    print("\tSUN397 total_images: {}".format(SUN397_total_images))
    print("\tpartition images: {}".format(len(SUN397_dataset)))
    print("\tclasses partition: {}".format(SUN397_dataset.classes_count))
    print("\tscenes classes hierarchy: {}".format(SUN397_dataset.classes_hierarchy))
    # SUN397_total_ver = 0
    # for count in SUN397_dataset.classes_count.values():
    #     SUN397_total_ver += count
    # print(SUN397_total_ver)

    print("Level0: other")
    


    
    

