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


## Final Dataset Metadata


### Classes hierarchy

The structure is:
- level0
    - level1
        - level2
            - level3
For every level0 the deepest sub level is the "last level"

Classes are:
- people
    - selfie
    - nonselfie
- scenes
    - indoor
        - sun397 level2 
            - sun397 level3
    - outdoor_natural
        - sun397 level2 
            - sun397 level3
    - outdoor_man-made
        - sun397 level2 
            - sun397 level3 
- other
    - pets
        - cat
            - 12 cats species
        - dog
            - 37 dogs species
    - cartoon
        - 100 famous people names
    - art
        - drawings
        - engraving
        - iconography
        - painting
        - sculpture

### CSV: Images 

It contains all the metada of the images that will present in the dataset

Fields:
- original_dataset: name of the original dataset
- img_folder: path to the image folder
- img_name: name of the image file
- split: if image belongs to 'train' or 'test' split
- level0: level0 class
- level1: level1 class
- level2: level2 class
- level3: level3 class
- last_level: since not every class have the same deep of hierarchy, in order to train the classifier with the deepest class level this field indicates which is

### CSV: Classes

It contains the hierarchy of different classes as explained below, their mapping into ids and their statistics

Fields:
- level0
- level0_id: level0 class id
- level1
- level1_id: level1 class id
- level2
- level2_id: level2 class id
- level3
- level3_id: level3 class id
- last_level
- last_level_id: last_level class id
- level0_#images
- level1_#images
- level2_#images
- level3_#images


La classe prevederà un campo target_type per estrarre rispettivamente come target il livello desiderato, "level0", "level1", "level2", "level3", "last_level"



## Selfie-Image-Detection-Dataset

### Statistiche

78619 images
{'selfie': 46836, 'nonselfie': 31783}
{'selfie': {'train': 37468, 'test': 9368}, 'nonselfie': {'train': 25426, 'test': 6357}}
image with smallest height: Test_data/Selfie/Selfie44658.jpg W=306 H=306
image with smallest width: Test_data/Selfie/Selfie44658.jpg W=306 H=306
Le immagini sono tutte della stessa dimensione 306x306, infatti sembra che alcune immagini siano state un po' deformate, soprattutto i nonselfie