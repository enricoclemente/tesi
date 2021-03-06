# From pytorch site: 
# "All datasets are subclasses of torch.utils.data.Dataset i.e, 
#  they have __getitem__ and __len__ methods implemented. 
#  Hence, they can all be passed to a torch.utils.data.DataLoader 
#  which can load multiple samples in parallel using 
#  torch.multiprocessing workers. "

# CIFAR10 class already exists in Pytorch but we need to create 
# a new version in order to apply transformation and when call __getitem__
# to return not the image and the label but the two transformed view of the image,
# i.e. the positive pairs

import torch
from PIL import Image
from torchvision.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


CIFAR10_normalization = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])


""" We create a subclass of CIFAR10 Pytorch class 
    torchvision.datasets.CIFAR10(root: str, 
                                train: bool = True, 
                                transform: Optional[Callable] = None, 
                                target_transform: Optional[Callable] = None, 
                                download: bool = False)
    
    We do that because we want custom __getitem__ method
"""
class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset modified, it will return two data augmentations for the same image
        This is needed for self-supervised contrastive learning
    """
    # for SSL we need a pair of two images from the dataset, 
    # in this case two transformation of the same image
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            if isinstance(self.transform, dict):
                # if augmentations are different
                augmentation_1 = self.transform['augmentation_1'](img)
                augmentation_2 = self.transform['augmentation_2'](img)
            else:
                augmentation_1 = self.transform(img)
                augmentation_2 = self.transform(img)
                
        
        # if index == 0:
        #     fig = plt.figure()
        #     plt.imshow(img)
        #     plt.savefig("original_img.png")
        #     fig = plt.figure()
        #     plt.imshow(im_1.numpy().transpose([1, 2, 0]))
        #     plt.savefig("transform_img1.png")
        #     fig = plt.figure()
        #     plt.imshow(im_2.numpy().transpose([1, 2, 0]))
        #     plt.savefig("transform_img2.png")

        

        return augmentation_1, augmentation_2
    
    # function to calculate mean and standard deviation of the dataset in order to normalize it
    def calculate_normalization_values(self):
        mean = np.round(self.data.mean(axis=(0,1,2))/255, 4)
        std = np.round(self.data.std(axis=(0,1,2))/255, 4)

        print("For CIFAR10 dataset we have mean {} and standard deviation {}".format(mean, std))
        return mean, std


""" We create a subclass of CIFAR10 Pytorch class 
    torchvision.datasets.CIFAR10(root: str, 
                                train: bool = True, 
                                transform: dictionary with two Optional[Callable]
                                            keys: 'augmentation_1', 'augmentation_2', 
                                target_transform: Optional[Callable] = None, 
                                download: bool = False)
    
    We do that because we want custom __getitem__ method
"""
class CIFAR10Triplet(CIFAR10):
    """CIFAR10 Dataset modified, it will return three images every time:
        - original image
        - trasformation1(image)
        - trasformation2(image)
    """
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        # if index == 0:
        #     fig = plt.figure()
        #     plt.imshow(img)
        #     plt.savefig("original_img.png")
        #     fig = plt.figure()
        #     plt.imshow(im_1.numpy().transpose([1, 2, 0]))
        #     plt.savefig("transform_img1.png")
        #     fig = plt.figure()
        #     plt.imshow(im_2.numpy().transpose([1, 2, 0]))
        #     plt.savefig("transform_img2.png")

        if self.transform is not None:
            augmentation_1 = self.transform['augmentation_1'](img)
            augmentation_2 = self.transform['augmentation_2'](img)
        
        # transform img to tensor for training
        img = transforms.Compose([transforms.ToTensor(),
                                CIFAR10_normalization])(img)
        
        return img, augmentation_1, augmentation_2

