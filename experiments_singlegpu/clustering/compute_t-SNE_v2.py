#!/usr/bin/env python
import sys
import os
import argparse
sys.path.insert(0, './')

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

import open_clip
from torchvision.models.resnet import resnet18
from experiments_singlegpu.datasets.SocialProfilePictures import SocialProfilePictures

import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from SPICE.spice.config import Config
from sklearn.manifold import TSNE




parser = argparse.ArgumentParser(description='UMAP+GMM calculator (experiments)')
parser.add_argument("--config_file", default="./experiments_config_example.py", metavar="FILE",
                    help="path to config file (same used with moco training)", type=str)
parser.add_argument('--dataset_folder', metavar='DIR', default='./datasets/cifar10',
                    help='path to dataset')
parser.add_argument('--save_folder', metavar='DIR', default='./results',
                    help='path to results')


def main():
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_file)

    # setting of save_folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    test_dataset = SocialProfilePictures(version=cfg.dataset.version, root=args.dataset_folder, split='test', randomize_metadata=cfg.dataset.randomize_metadata,
                    transform=transforms.Compose([  transforms.Resize([cfg.dataset.img_size, cfg.dataset.img_size]), 
                                                    transforms.ToTensor()]))

    if cfg.model == "resnet18":
        model = resnet18()
        if not os.path.isfile(cfg.model_path):
            # if not available a previously trained model on moco use pretrained one on Imagenet
            print("Using pretrained model on ImageNet")
            model = resnet18(pretrained=True)
        else:
            model = resnet18(num_classes=128)
            loc = 'cuda:{}'.format(torch.cuda.current_device())
            checkpoint = torch.load(cfg.model_path, map_location=loc)
            dim_mlp = model.fc.in_features
            model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
            state_dict = dict()
            for key in checkpoint['state_dict']:
                
                if key.startswith("encoder_q"):
                    # print(key[22:])
                    state_dict[key[10:]] = checkpoint['state_dict'][key]
            model.load_state_dict(state_dict, strict=False)

        if cfg.features_layer == 'layer4':
            # removing fc layer 
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
    
    elif cfg.model == "vit_b_32":
        if not os.path.isfile(cfg.model_path):
            # if not available a previously trained model on moco use pretrained one on Imagenet
            print("Using pretrained model on LAION-2B E16")
            model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_e16')
        else:
            raise NotImplementedError("Not yet implemented trained ViT model")
    else:
        raise NotImplementedError("Choose a valid model!")
    

    
    for p in model.parameters():
        p.requires_grad = False

    # print(model)
    model.cuda()

    print("Extracting test features")
    test_features, test_targets = extract_features_targets(model, test_dataset, cfg)

    print("Calculating t-SNE")
    tsne_2d = TSNE(n_components=2).fit_transform(test_features)
    plot_tSNE(test_dataset, tsne_2d, test_targets, args.save_folder)


def extract_features_targets(model, dataset, cfg):
    features = []
    targets = []

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1, 
                pin_memory=True, drop_last=False)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(loader):
            img = img.cuda()

            if cfg.model == "resnet18":
                feature = model.forward(img)
            elif cfg.model == "vit_b_32":
                feature = model.encode_image(img)
            
            if cfg.features_layer == 'layer4' and cfg.model == "resnet18":
                # flattening and normalizing if extracting from non fc layers
                feature = torch.flatten(feature, 1)
            
            feature = F.normalize(feature, dim=1)

            # collecting all features and targets
            features.append(feature.cpu())
            targets.append(target)
            
            print("[{}]/[{}] batch iteration".format(i, len(loader)))

    features = torch.cat(features)
    features = features.numpy()

    targets = torch.cat(targets)
    targets = targets.numpy()

    return features, targets


def plot_tSNE(dataset, embedding, targets, save_folder):
    colors_per_class = {}
    for class_name in dataset.classes:
        colors_per_class[class_name] = list(np.random.choice(range(256), size=3))
    
    # extract x and y coordinates 
    tx = embedding[:, 0]
    ty = embedding[:, 1]

    # initialize a matplotlib plot
    fig = plt.figure("t-SNE", figsize=(20,15))
    ax = fig.add_subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    plots = list(range(len(dataset.classes)))
    # for every class, we'll add a scatter plot separately
    for class_index, class_name in enumerate(dataset.classes):
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(targets) if l == class_index]
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        # convert the class color to matplotlib format
        color = np.array(colors_per_class[class_name], dtype=float) / 255
        # add a scatter plot with the corresponding color and label
        plots[class_index] = ax.scatter(current_tx, current_ty,  c=color, label=class_name)
    
    # build a legend using the labels we set previously
    ax.legend(bbox_to_anchor=(1.0, 1.0))
    legendFig = plt.figure("t-SNE legend", figsize=(5,7))
    legendFig.legend(plots, dataset.classes, loc='center')

    # finally, show the plot
    fig.savefig('{}/t-SNE.svg'.format(save_folder))
    # legendFig.savefig('{}/UMAP_legend.svg'.format(args.save_folder))
    plt.close()


if __name__ == '__main__':
    main()
