# 2nd Phase for SPICE: precompute embedding features

In this phase we have to extract features from the self-supervised model. 
In the case of MoCo we have to use only the query encoder removing the avgpool and the fc layer in order to obtain features

## Script
pre_compute_embedding.py

### Description
Takes the MoCo query encoder with loaded weights, removes the avgpool and fc layer and use it to extract features from the dataset. It saves the list with all the features in save_folder

### Arguments
- dataset_folder: path to dataset
- model_path: path to pretrained model
- save_folder: path where to save features
- logs_folder: now it's useless
- batch-size: number of images per iteration in the dataset
- moco-dim: moco features dimension, 
- moco-k: moco queue size
- moco-m: moco momentum for updating key encoder
- moco-t: moco softmax temperature
- mlp: if moco will use mlp


