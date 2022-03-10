import os
import collections

import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Union, Tuple
import scipy.io
from PIL import Image
import re


"""
    IIIT Cartoon Faces in the Wild (IIT-CFW) dataset implementation for pytorch
    site: https://cvit.iiit.ac.in/research/projects/cvit-projects/cartoonfaces
    paper: https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2016/Mishra-ECCVW2016.pdf

    The dataset contains cartoon faces of 100 famous people in different styles

    Labels are not hierarchical but same images have different labels: 
        - name of the famous person (class)
        - gender
        - age
        - glass
        - beard
        (- type of cartoon
        - pose
        - expression) --> didn't find them
        - face bounding box (we are not interested)
    
    Statistics:
        8928 images
        {'aamirkhan': 42, 'abrahamlincoln': 176, 'adolfhitler': 82, 'aishwaryarai': 105, 'alberteinstein': 201, 'amitabhbachan': 50, 'angelamerkel': 57, 'angelinajolie': 299, 'arnoldschwazegger': 89, 'barackobama': 262, 'beyonce': 45, 'bhagatsingh': 53, 'billclinton': 59, 'billgates': 66, 'bradpitt': 133, 'britneyspears': 93, 'brucelee': 168, 'brucewillis': 122, 'charliechaplin': 131, 'cheguevara': 105, 'curtisjamesjacksoniii': 20, 'dalailama': 182, 'danielcraig': 93, 'danielradicliffe': 75, 'davidbeckham': 79, 'dwaynejohnson': 80, 'elvispresley': 88, 'emmawatson': 249, 'georgeclooney': 89, 'hillaryclinton': 121, 'hughjackman': 198, 'indiragandhi': 31, 'jkrowling': 13, 'jackiechan': 57, 'jawaharlalnehru': 22, 'jay-z': 36, 'jimcarrey': 86, 'johnfkennedy': 61, 'johnlennon': 127, 'johnnydepp': 269, 'justinbieber': 171, 'katemiddleton': 58, 'katrinakaif': 35, 'kimjongun': 60, 'laluprasadyadav': 75, 'lancearmstrong': 25, 'latamangeshkar': 13, 'leonardodicaprio': 143, 'lionelmessi': 123, 'louisarmstrong': 51, 'lucianopavarotti': 41, 'lucilleball': 51, 'mahatmagandhi': 104, 'malcolmx': 12, 'manmohansingh': 114, 'margaretthatcher': 51, 'marilynmonroe': 199, 'markzuckerberg': 53, 'martinlutherking': 144, 'mattdamon': 57, 'meganfox': 142, 'michaeljackson': 124, 'michaeljordan': 71, 'michaelphelps': 32, 'morganfreeman': 104, 'motherteresa': 132, 'muhammadali': 55, 'narendramodi': 198, 'nelsonmandela': 163, 'nicolekidman': 44, 'oprahwinfrey': 42, 'pablopicasso': 42, 'paulmccartney': 67, 'pele': 21, 'peterjackson': 11, 'petersellers': 16, 'popejohnpaulii': 52, 'princecharles': 51, 'princessdiana': 71, 'queenelizabethii': 53, 'quentintarantino': 61, 'rafaelnadal': 48, 'rajinikanth': 53, 'robertdowneyjr': 100, 'robynrihannafenty': 62, 'rowanatkinson': 86, 'russellcrowe': 54, 'sachintendulkar': 59, 'scarlettjohansson': 119, 'selenagomez': 85, 'shakira': 80, 'stephenhawking': 40, 'stevejobs': 112, 'sylvesterstallone': 86, 'taylorswift': 113, 'tigerwoods': 42, 'tomcruise': 76, 'usainbolt': 72, 'vladimirputin': 160, 'winstonchurchill': 35}
        The two most smallest images are: 
            with the smallest H: cartoonFaces/NarendraModi0188.jpeg W= 32 H= 21 
            with the smallest W: cartoonFaces/IndiraGandhi0005.jpeg W= 21 H= 24

"""
class CFW(Dataset):
    """
        Selfie-Image-Detection-Dataset Dataset

        Args: 
            root (string): Root directory where images are downloaded to or better to the extracted sun397 folder.
                            folder data structure should be:
                            data (root)
                                |-cartoonFaces
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
        # annotation file is in .mat format. A structured file 
        annotations_file = scipy.io.loadmat(os.path.join(self.root, 'IIIT-CFW1.0','IIITCFWdata.mat'))

        # create map of classes { classname: index }
        classes_map = {}

        # create map of other attributes
        attributes_map = {  'gender': { 'male': 0, 'female': 1},
                            'age': { 'young': 0, 'old': 1},
                            'glass': { 'yes': 0, 'no': 1},
                            'beard': { 'yes': 0, 'no': 1},
                        }
                        
        # create map of classes with number of images per classes { classname: # of images in class }
        classes_count = {}
        # will be used to distribute in the same proportion each class into ttrain and test
        classes_splitter = {}

        # structures for tracking filtered images due to thresholds
        filtered_classes_count = {}
        total_filtered = 0

        # will be used to distribute equally images among classes
        distributed_classes_count = {}

        classes_index = 0
        # make statistics on the dataset
        for i in range(len(annotations_file['IIITCFWdata'][0][0][1][0])):
            # extract from img_name the classname
            img_path = annotations_file['IIITCFWdata'][0][0][1][0][i][0]
            class_name = re.split(r'(\d+)', annotations_file['IIITCFWdata'][0][0][1][0][i][0].split('/')[1].split('.')[0])[0].lower()
            if class_name not in classes_map:
                classes_map[class_name] = classes_index
                classes_index += 1
                classes_count[class_name] = 1
                classes_splitter[class_name] = 0
                filtered_classes_count[class_name] = 1
                distributed_classes_count[class_name] = 0
            else:
                classes_count[class_name] += 1
                filtered_classes_count[class_name] += 1
            
            total_images += 1

            W, H = Image.open(os.path.join(self.root, 'IIIT-CFW1.0', img_path)).size

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
        
        # print(classes_map)
        # print(classes_count)
        # print(classes_splitter) 
        for i in range(len(annotations_file['IIITCFWdata'][0][0][1][0])):
            img_path = annotations_file['IIITCFWdata'][0][0][1][0][i][0]
            class_name = re.split(r'(\d+)', annotations_file['IIITCFWdata'][0][0][1][0][i][0].split('/')[1].split('.')[0])[0].lower()
            gender = annotations_file['IIITCFWdata'][0][0][2][0][i][0]
            age = annotations_file['IIITCFWdata'][0][0][3][0][i][0]
            glass = annotations_file['IIITCFWdata'][0][0][4][0][i][0]
            beard = annotations_file['IIITCFWdata'][0][0][5][0][i][0]

            W, H = Image.open(os.path.join(self.root, 'IIIT-CFW1.0', img_path)).size
            
            skip = False
            if self.aspect_ratio_threshold is not None and (W/H > self.aspect_ratio_threshold or W/H < 1/self.aspect_ratio_threshold):
                skip = True
            elif self.area_threshold is not None and (W*H < self.area_threshold):
                skip = True
            if skip == False:
                meta = {}
                if 'train' in self.split:
                    if classes_splitter[class_name] < int(filtered_classes_count[class_name] * self.split_perc):
                        meta['split'] = 'train'
                        meta['img_name'] = img_path.split('/')[1]
                        meta['img_folder'] = os.path.join('IIIT-CFW1.0', img_path.split('/')[0])
                        meta['target'] = { 'level1': 'cartoon',
                                            'level1_attributes': { 'gender': gender,
                                                                    'age': age,
                                                                    'glass': glass,
                                                                    'beard': beard},
                                            'level2': class_name,
                                            'level3': class_name,
                                        }
                        targets.append(meta['target'])
                        metadata.append(meta)
                        classes_splitter[class_name] += 1
                    elif (classes_splitter[class_name] >= int(filtered_classes_count[class_name] * self.split_perc) 
                        and classes_splitter[class_name] < int(filtered_classes_count[class_name])):
                        if "test" in self.split:
                            meta['split'] = 'test'
                            meta['img_name'] = img_path.split('/')[1]
                            meta['img_folder'] = os.path.join('IIIT-CFW1.0', img_path.split('/')[0])
                            meta['target'] = { 'level1': 'cartoon',
                                                'level1_attributes': { 'gender': gender,
                                                                        'age': age,
                                                                        'glass': glass,
                                                                        'beard': beard},
                                                'level2': class_name,
                                                'level3': class_name,
                                            }
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

        return (metadata, targets, classes_map, classes_splitter, 
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
