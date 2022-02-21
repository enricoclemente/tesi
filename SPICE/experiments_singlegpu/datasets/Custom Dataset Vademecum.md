# Practical Guide to Custom Dataset

This guide lead to understand how to create custom dataset in order to use same notations and standards as pytorch does for torchvision.datasets classes

## Standard attributes
- classes = list of class names (strings)
- classes_map = map of classes with relative indices {'class name': index }
- classes_count = map of classes with relative number of images {'class name': # of images }
- targets = list of labels for every image in the dataset; often ints since you will use classifier to predict a number for the relative class.

- metada:
    - split
    - img_name
    - img_folder
    - target