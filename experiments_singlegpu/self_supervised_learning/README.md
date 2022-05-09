# Self-Supervised Learning

This folder contains scripts useful to make training and evaluation of datasets on MoCo v2

## train_moco.py 

This script execute the training of MoCo v2 framework, the starting code is taken from official repository https://github.com/facebookresearch/moco. 
Then the code has been readapted for my experiments. Several parts have been added like knn validation, linear classifier validation.

### arguments
- config_file: path to 