import os
import shutil
import csv


import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Union, Tuple
import scipy.io
from PIL import Image
import PIL
import matplotlib.pyplot as plt

from experiments_singlegpu.datasets.SelfieImageDetectionDataset_custom import SIDD
from experiments_singlegpu.datasets.SUN397_custom import SUN397
from experiments_singlegpu.datasets.OxfordIIITPet_custom import OxfordIIITPet
from experiments_singlegpu.datasets.IIITCFW_custom import CFW
from experiments_singlegpu.datasets.ArtImages_custom import ArtImages
from experiments_singlegpu.datasets.iCartoonFace_custom import iCartoonFace


# indicates level0 classes and relative percentual presence in the final dataset
level0_classes = {  'people': { 'perc': 0.6 },
                    'scenes': { 'perc': 0.2 },
                    'other': { 'perc': 0.2 }}

dataset_colors = {'SIDD_dataset_filtered_partition': 'blue', 
                 'SUN397_dataset_filtered_partition': 'red', 
                 'Oxford_dataset_filtered_partition': 'green',
                 'Cartoon_dataset_filtered_partition': 'pink', 
                 'ArtImages_dataset_filtered_partition': 'purple'}

dataset_legend = {'People': 'blue', "Scenes": 'red', "Pets": 'green', "Cartoon": "pink", 'Art': 'purple'}

"""
    Create Images CSV of the final dataset starting from the datasets
    Args:
        - total_images: the number of total images of the final dataset
        - split_perc: the percetange of split between train images and test images
        - aspect_ratio_threshold: floating value indicating the trheshold to filter images that are 
                                    over a certain aspect ratio
        - dim_threshold: integer value indicating min dimension threshold to filter images that have area
                        smaller than dim_threshold * dim_threshold * 1/aspect_ratio_threshold
                        example: 
                            aspect_ratio_threshold = 2.33 (21:9)
                            dim_threshold = 225 (px) means images that have area small than 225x96(dim_threshold * 1/aspect_ratio_threshold)
        - images_csv_path: string indicating path where to save images_metadata.csv
    
    The images in the classes will keep the same proportions as from the original statistics
"""
def create_images_csv(total_images=60000, split_perc=0.8, aspect_ratio_threshold=2.33, dim_threshold=225, images_csv_path='./'):
    # First open the datasets and gets overall statistics

    # Working on level0: people
    print("Level0: people")
    # Take a partition of SIDD dataset == total_images * people perc
    SIDD_dataset_original = SIDD(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/Selfie-Image-Detection-Dataset/data',
                split=["train", "test"], split_perc=split_perc)
    SIDD_dataset_filtered = SIDD(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/Selfie-Image-Detection-Dataset/data',
                    split=["train", "test"], split_perc=split_perc, 
                    aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    SIDD_partition_perc = level0_classes['people']['perc'] * total_images / len(SIDD_dataset_filtered)
    SIDD_dataset_filtered_partition = SIDD(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/Selfie-Image-Detection-Dataset/data',
                    split=["train", "test"], split_perc=split_perc, partition_perc=SIDD_partition_perc, distribute_images=True,
                    aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    
    print("\tSIDD dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(SIDD_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(SIDD_dataset_filtered), SIDD_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(SIDD_dataset_filtered_partition), SIDD_partition_perc))
    print("\t\tclasses partition: {}".format(SIDD_dataset_filtered_partition.classes_count))
    print("\t\tclasses filtering ratio: {}".format(SIDD_dataset_filtered_partition.filtering_classes_effect))

    # Working on level0: scenes
    print("Level0: scenes")
    SUN397_dataset_original = SUN397(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/SUN397/data', 
                    split=["train","test"], split_perc=split_perc)
    SUN397_dataset_filtered = SUN397(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/SUN397/data', 
                        split=["train","test"], split_perc=split_perc, 
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    SUN397_partition_perc = level0_classes['scenes']['perc'] * total_images / len(SUN397_dataset_filtered)
    SUN397_dataset_filtered_partition = SUN397(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/SUN397/data', 
                        split=["train","test"], split_perc=split_perc, partition_perc=SUN397_partition_perc, distribute_images=True, distribute_level='level2',
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    
    
    print("\tSUN397 dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(SUN397_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(SUN397_dataset_filtered), SUN397_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(SUN397_dataset_filtered_partition), SUN397_partition_perc))
    print("\t\tclasses partition: {}".format(SUN397_dataset_filtered_partition.classes_count))
    print("\t\tscenes classes hierarchy: {}".format(SUN397_dataset_filtered_partition.classes_hierarchy))
    print("\t\tclasses filtering ratio: {}".format(SUN397_dataset_filtered_partition.filtering_classes_effect))
    # SUN397_total_ver = 0
    # for count in SUN397_dataset.classes_count.values():
    #     SUN397_total_ver += count
    # print(SUN397_total_ver)
    
    print("Level0: other")
    print("Level1: Pets")
    Oxford_dataset_original = OxfordIIITPet(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/OxfordIII-TPet/data', 
                    split=["train","test"], split_perc=split_perc)
    Oxford_dataset_filtered = OxfordIIITPet(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/OxfordIII-TPet/data', 
                        split=["train","test"], split_perc=split_perc, 
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    Oxford_partition_perc = level0_classes['other']['perc']/3 * total_images / len(Oxford_dataset_filtered)
    Oxford_dataset_filtered_partition = OxfordIIITPet(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/OxfordIII-TPet/data', 
                        split=["train","test"], split_perc=split_perc, partition_perc=Oxford_partition_perc, distribute_images=True, distribute_level='level2',
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    print("\tOxfordIII-Tpet dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(Oxford_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(Oxford_dataset_filtered), Oxford_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(Oxford_dataset_filtered_partition), Oxford_partition_perc))
    print("\t\tclasses partition: {}".format(Oxford_dataset_filtered_partition.classes_count))
    print("\t\tscenes classes hierarchy: {}".format(Oxford_dataset_filtered_partition.classes_hierarchy))
    print("\t\tclasses filtering ratio: {}".format(Oxford_dataset_filtered_partition.filtering_classes_effect))
    
    print("Level1: Cartoon")
    Cartoon_dataset_original = iCartoonFace(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/iCartoonFace/data', 
                    split=["train","test"], split_perc=split_perc)
    Cartoon_dataset_filtered = iCartoonFace(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/iCartoonFace/data', 
                        split=["train","test"], split_perc=split_perc, 
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    Cartoon_partition_perc = level0_classes['other']['perc']/3 * total_images / len(Cartoon_dataset_filtered)
    Cartoon_dataset_filtered_partition = iCartoonFace(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/iCartoonFace/data', 
                        split=["train","test"], split_perc=split_perc, partition_perc=Cartoon_partition_perc, distribute_images=True,
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    print("\tiCartoonFace dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(Cartoon_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(Cartoon_dataset_filtered), Cartoon_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(Cartoon_dataset_filtered_partition), Cartoon_partition_perc))
    print("\t\tclasses partition: {}".format(Cartoon_dataset_filtered_partition.classes_count))
    print("\t\tclasses filtering ratio: {}".format(Cartoon_dataset_filtered_partition.filtering_classes_effect))
    # CFW_dataset_original = CFW(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/IIIT-CFW/data', 
    #                 split=["train","test"], split_perc=split_perc)
    # CFW_dataset_filtered = CFW(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/IIIT-CFW/data', 
    #                     split=["train","test"], split_perc=split_perc, 
    #                     aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    # CFW_partition_perc = level0_classes['other']['perc']/3 * total_images / len(CFW_dataset_filtered)
    # if CFW_partition_perc > 1.0:
    #     print("CFW database has not enough images!")
    #     CFW_partition_perc = 0.99
    # CFW_dataset_filtered_partition = CFW(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/IIIT-CFW/data', 
    #                     split=["train","test"], split_perc=split_perc, partition_perc=CFW_partition_perc, distribute_images=True,
    #                     aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    # print("\tIIITCFW dataset statistics")
    # print("\t\toriginal dataset images: {}".format(len(CFW_dataset_original)))
    # print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(CFW_dataset_filtered), CFW_dataset_filtered.total_filtered))
    # print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(CFW_dataset_filtered_partition), CFW_partition_perc))
    # print("\t\tclasses partition: {}".format(CFW_dataset_filtered_partition.classes_count))
    # print("\t\tclasses filtering ratio: {}".format(CFW_dataset_filtered_partition.filtering_classes_effect))
    

    print("Level1: Art")
    ArtImages_dataset_original = ArtImages(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/ArtImages/data', 
                    split=["train","test"], split_perc=split_perc)
    ArtImages_dataset_filtered = ArtImages(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/ArtImages/data', 
                        split=["train","test"], split_perc=split_perc, 
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    ArtImages_partition_perc = level0_classes['other']['perc']/3 * total_images / len(ArtImages_dataset_filtered)
    ArtImages_dataset_filtered_partition = ArtImages(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/ArtImages/data', 
                        split=["train","test"], split_perc=split_perc, partition_perc=ArtImages_partition_perc, distribute_images=True,
                        aspect_ratio_threshold=aspect_ratio_threshold, dim_threshold=dim_threshold)
    print("\tArtImages dataset statistics")
    print("\t\toriginal dataset images: {}".format(len(ArtImages_dataset_original)))
    print("\t\tfiltered dataset images: {} (filtered images: {})".format(len(ArtImages_dataset_filtered), ArtImages_dataset_filtered.total_filtered))
    print("\t\tfiltered partition dataset images: {} (partition percentage: {}%)".format(len(ArtImages_dataset_filtered_partition), ArtImages_partition_perc))
    print("\t\tclasses partition: {}".format(ArtImages_dataset_filtered_partition.classes_count))
    print("\t\tclasses filtering ratio: {}".format(ArtImages_dataset_filtered_partition.filtering_classes_effect))

   
    final_dataset_level_1_classes_count = {}
    final_dataset_level_2_classes_count = {}
    final_dataset_level_3_classes_count = {}
    for d in [SIDD_dataset_filtered_partition, SUN397_dataset_filtered_partition, Oxford_dataset_filtered_partition,
                Cartoon_dataset_filtered_partition, ArtImages_dataset_filtered_partition]:
        for t in d.targets:
            if t['level1'] not in final_dataset_level_1_classes_count:
                final_dataset_level_1_classes_count[t['level1']] = 1
            elif t['level1'] in final_dataset_level_1_classes_count:
                final_dataset_level_1_classes_count[t['level1']] += 1

            if t['level2'] not in final_dataset_level_2_classes_count:
                final_dataset_level_2_classes_count[t['level2']] = 1
            elif t['level2'] in final_dataset_level_2_classes_count:
                final_dataset_level_2_classes_count[t['level2']] += 1

            if t['level3'] not in final_dataset_level_3_classes_count:
                final_dataset_level_3_classes_count[t['level3']] = 1
            elif t['level3'] in final_dataset_level_3_classes_count:
                final_dataset_level_3_classes_count[t['level3']] += 1

    hist_elements = 30
    # print(final_dataset_level_2_classes_count)    
    final_dataset_level_1_classes_count_sorted = {}
    final_dataset_level_1_classes_color = []
    for key, value in sorted(final_dataset_level_1_classes_count.items(), key=lambda item: item[1]):
        final_dataset_level_1_classes_count_sorted[key] = value
        for i, d in enumerate([SIDD_dataset_filtered_partition, SUN397_dataset_filtered_partition, Oxford_dataset_filtered_partition,
                Cartoon_dataset_filtered_partition, ArtImages_dataset_filtered_partition]):
            if key in d.classes_map['level1']:
                final_dataset_level_1_classes_color.append(list(dataset_colors.values())[i])

    print("Level 1 Classes Distribution: {}".format(final_dataset_level_1_classes_count_sorted))
    plt.xticks(rotation=45, ha='right')
    plt.bar(list(final_dataset_level_1_classes_count_sorted.keys()), list(final_dataset_level_1_classes_count_sorted.values()), color=final_dataset_level_1_classes_color)
    plt.gca().set(title='Level 1 Partitioned Filtered Classes Distribution', ylabel='# of images')
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(list(final_dataset_level_1_classes_count_sorted.values())):
        plt.text(xlocs[i] - 0.3, v + 0.1, str(v), fontsize=8)
    for i, j in dataset_legend.items(): #Loop over color dictionary
        plt.bar(i, 0,width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("experiments_singlegpu/datasets/final_dataset_filtered_partition_level1_classes_distribution.svg")
    plt.close()

    final_dataset_level_2_classes_count_sorted = {}
    final_dataset_level_2_classes_color = []
    for key, value in sorted(final_dataset_level_2_classes_count.items(), key=lambda item: item[1]):
        final_dataset_level_2_classes_count_sorted[key] = value
        for i, d in enumerate([SIDD_dataset_filtered_partition, SUN397_dataset_filtered_partition, Oxford_dataset_filtered_partition,
                Cartoon_dataset_filtered_partition, ArtImages_dataset_filtered_partition]):
            if key in d.classes_map['level2']:
                final_dataset_level_2_classes_color.append(list(dataset_colors.values())[i])
    print("Level 2 Classes Distribution: {}".format(final_dataset_level_2_classes_count_sorted))
    plt.xticks(rotation=45, ha='right')
    plt.bar(list(final_dataset_level_2_classes_count_sorted.keys()), list(final_dataset_level_2_classes_count_sorted.values()), color=final_dataset_level_2_classes_color)
    plt.gca().set(title='Level 2 Partitioned Filtered Classes Distribution', ylabel='# of images')
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(list(final_dataset_level_2_classes_count_sorted.values())):
        plt.text(xlocs[i] - 0.3, v + 0.1, str(v), fontsize=4)
    for i, j in dataset_legend.items(): #Loop over color dictionary
        plt.bar(i, 0,width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("experiments_singlegpu/datasets/final_dataset_filtered_partition_level2_classes_distribution.svg")
    plt.close()
    
    final_dataset_level_3_classes_count_sorted = {}
    final_dataset_level_3_classes_color = []
    for key, value in sorted(final_dataset_level_3_classes_count.items(), key=lambda item: item[1]):
        final_dataset_level_3_classes_count_sorted[key] = value
        for i, d in enumerate([SIDD_dataset_filtered_partition, SUN397_dataset_filtered_partition, Oxford_dataset_filtered_partition,
                Cartoon_dataset_filtered_partition, ArtImages_dataset_filtered_partition]):
            if key in d.classes_map['level3']:
                final_dataset_level_3_classes_color.append(list(dataset_colors.values())[i])
    print("Level 3 Classes Distribution: {}".format(final_dataset_level_3_classes_count_sorted))
    plt.xticks(rotation=45, ha='right')
    plt.bar(list(final_dataset_level_3_classes_count_sorted.keys())[:hist_elements], list(final_dataset_level_3_classes_count_sorted.values())[:hist_elements], color=final_dataset_level_3_classes_color[:hist_elements])
    plt.gca().set(title='Level 3 Partitioned Filtered Classes Distribution', ylabel='# of images')
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(list(final_dataset_level_3_classes_count_sorted.values())[:hist_elements]):
        plt.text(xlocs[i] - 0.3, v + 0.1, str(v), fontsize=4)
    for i, j in dataset_legend.items(): #Loop over color dictionary
        plt.bar(i, 0,width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("experiments_singlegpu/datasets/final_dataset_filtered_partition_level3_classes_distribution.svg")
    plt.close()


    # creation of the images csv
    with open(os.path.join(images_csv_path + 'images_metadata.csv'), mode='w') as file:
        csv_writer = csv.writer(file, delimiter=',')
        csv_writer.writerow(['original_dataset', 'img_folder', 'img_name', 'split', 'level0', 'level1', 'level2', 'level3', 'target_level'])

        for m in SIDD_dataset_filtered_partition.metadata:
            csv_writer.writerow(['Selfie-Image-Detection-Dataset', m['img_folder'], m['img_name'], m['split'], 'people', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level3']])
        for m in SUN397_dataset_filtered_partition.metadata:
            csv_writer.writerow(['SUN397', m['img_folder'], m['img_name'], m['split'], 'scenes', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level2']])
        for m in Oxford_dataset_filtered_partition.metadata:
            csv_writer.writerow(['OxfordIII-TPet', m['img_folder'], m['img_name'], m['split'], 'other', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level2']])
        for m in Cartoon_dataset_filtered_partition.metadata:
            csv_writer.writerow(['iCartoonFace', m['img_folder'], m['img_name'], m['split'], 'other', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level3']])
        for m in ArtImages_dataset_filtered_partition.metadata:
            csv_writer.writerow(['ArtImages', m['img_folder'], m['img_name'], m['split'], 'other', m['target']['level1'], m['target']['level2'], m['target']['level3'], m['target']['level3']])




"""
    Social Profile Pictures Dataset

    The dataset contains is a composite one, it is created starting from several datasets in order to reproduce a
    real life scenario of social media profile pictures like Facebook Profile Pictures

    Labels:
    
    Statistics:

"""
class SocialProfilePictures(Dataset):
    """
    Selfie-Image-Detection-Dataset Dataset

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
        split (string or list): possible options: 'train', 'test', 
            if list of multiple splits they will be treated as unique split
        split_perc (float): in order to custom the dataset you can choose the split percentage
        transform (callable, optional): A function/transform that  takes in an PIL image 
            and returns a transformed version. E.g, ``transforms.ToTensor``
        partition_perc (float): use it to take only a part of the dataset, keeping the proportion of number of images per classes
            split_perc will work as well splitting the partion
        aspect_ratio_threshold (float): use it to filter images that have a greater aspect ratio
        dim_threshold (float): use it to filter images which area is 
            lower of = dim_threshold * dim_threshold * 1/aspect_ratio_threshold
    """
    def __init__(self, root: str, split: Union[List[str], str] = "train", split_perc: float = 0.8,
                transform: Optional[Callable] = None, partition_perc: float = 1.0,
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

        if aspect_ratio_threshold is not None:
            self.aspect_ratio_threshold = aspect_ratio_threshold
        else:
            self.aspect_ratio_threshold = None

        if dim_threshold is not None:
            self.area_threshold = dim_threshold * dim_threshold * 1/aspect_ratio_threshold
        else: 
            self.area_threshold = None
        
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
        classes_count = {}

        # calculating statistics
        with open(os.path.join(self.root, 'SocialProfilePictures', 'data', 'images_metadata.csv')) as file:
            csv_reader = csv.reader(file, delimiter=',')
            line_count = 0

            class_index = 0
            for row in csv_reader:
                if line_count == 0:
                    # ignore columns
                    line_count += 1
                else:
                    if row[8] not in classes_map:
                        classes_map[row[8]] = class_index
                        class_index += 1
                        classes_count[row[8]] = 1
                    else: 
                        classes_count[row[8]] += 1
                    line_count += 1
            print(line_count)
            
        print(classes_map)
        print(classes_count)

        with open(os.path.join(self.root, 'SocialProfilePictures', 'data', 'images_metadata.csv')) as file:
            csv_reader = csv.reader(file, delimiter=',')
            line_count = 0

            for row in csv_reader:
                if line_count == 0:
                    # ignore columns
                    line_count += 1
                else:
                    meta = {}
                    if "train" in self.split and row[3] == 'train':
                        meta['split'] = row[3]
                        meta['img_name'] = row[2]
                        meta['img_folder'] = os.path.join(row[0], 'data', row[1])
                        meta['levels'] = {'level1': row[5], 'level2': row[6], 'level3': row[7], 'target_level': row[8]}
                        targets.append(classes_map[row[8]])
                        metadata.append(meta)
                    if "test" in self.split and row[3] == 'test':
                        meta['split'] = row[3]
                        meta['img_name'] = row[2]
                        meta['img_folder'] = os.path.join(row[0], 'data', row[1])
                        meta['levels'] = {'level1': row[5], 'level2': row[6], 'level3': row[7], 'target_level': row[8]}
                        targets.append(classes_map[row[8]])
                        metadata.append(meta)

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










    
    

