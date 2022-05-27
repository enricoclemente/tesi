#!/usr/bin/env python
import argparse
import os
import sys
sys.path.insert(0, './')

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from SPICE.spice.config import Config
from SPICE.spice.model.feature_modules.resnet_cifar import resnet18_cifar
from torchvision.models import resnet18

from torchvision.datasets import CIFAR10
from experiments_singlegpu.datasets.SocialProfilePictures import SocialProfilePictures

from experiments_singlegpu.datasets.utils.custom_transforms import PadToSquare


parser = argparse.ArgumentParser(description='Tensorboard Projector Creator')
parser.add_argument("--config_file", default="./experiments_config_example.py", metavar="FILE",
                    help="path to config file", type=str)
parser.add_argument('--dataset', default="cifar10", type=str,
                    help="name of the dataset, this lead to different script choices")
parser.add_argument('--dataset_folder', metavar='DIR', default='./datasets/cifar10',
                    help='path to dataset')
parser.add_argument('--model_path', default=None, type=str, metavar='PATH',
                    help='path to previously trained model, if not set and dataset is SPP, it will be used ResNet18 pretrained on ImageNet')
parser.add_argument('--logs_folder', metavar='DIR', default='./results/cifar10/moco/logs',
                    help='path to tensorboard logs')
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')


def main():
    args = parser.parse_args()
    print(args)
    cfg = Config.fromfile(args.config_file)

    # setting of logs_folder
    if not os.path.exists(args.logs_folder):
        os.makedirs(args.logs_folder)

    # checking GPU and showing infos
    if not torch.cuda.is_available():
        raise NotImplementedError("You need GPU!")
    print("Use GPU: {} for training".format(torch.cuda.current_device()))

    torch.cuda.set_device(torch.cuda.current_device())

    model = None
    dataset = None
    dataset_sprites = None

    dataset_normalization = transforms.Normalize(mean=cfg.dataset.normalization.mean, std=cfg.dataset.normalization.std)
    if args.dataset == 'cifar10':
        # creating model MoCo using resnet18_cifar which is an implementation adapted for CIFAR10
        model = resnet18_cifar()

        dataset = CIFAR10(root=args.dataset_folder, train=False, 
                transform=transforms.Compose([transforms.ToTensor()]),  
                download=True)
    elif args.dataset == 'socialprofilepictures':
        model = resnet18(pretrained=True if not args.model_path else False)

        dataset = SocialProfilePictures(version=cfg.dataset.version, root=args.dataset_folder, split='val', 
                transform=transforms.Compose([ PadToSquare(),
                                                transforms.Resize([cfg.dataset.img_size, cfg.dataset.img_size]), 
                                                transforms.ToTensor()]))
        dataset_sprites = SocialProfilePictures(version=cfg.dataset.version, root=args.dataset_folder, split='val', 
                transform=transforms.Compose([ PadToSquare(), 
                                                transforms.Resize([64, 64]), 
                                                transforms.ToTensor()]))
        print("Dataset is big: {} images".format(len(dataset)))
        print("Classes distribution: {}".format(dataset.classes_count))
        exit()
    else:
        raise NotImplementedError("Choose a valid dataset!")

    # optionally resume from a checkpoint
    if args.model_path:
        if os.path.isfile(args.model_path):
            print("=> loading model parameters '{}'".format(args.model_path))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(torch.cuda.current_device())
            checkpoint = torch.load(args.model_path, map_location=loc)
            prev_parameteres = model.parameters()
            state_dict = dict()
            for key in checkpoint['state_dict']:
                print(key)
                if key.startswith("encoder_q"):
                    state_dict[key[10:]] = checkpoint['state_dict'][key]
            
            model.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError("=> no checkpoint found at '{}'".format(args.model_path))
    else:
        print("Model path not specified, if the dataset is SPP, pretrained model on ImageNet will be used")


    # remove fc from encoder in order to get features
    model = torch.nn.Sequential(*(list(model.children())[:-1])).cuda()
    print(model)
    cudnn.benchmark = True  # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

    # tensorboard plotter
    test_writer = SummaryWriter(args.logs_folder + "/validation_set")

    create_projector(model, dataset, dataset_sprites, test_writer, 'layer_4', args)

    
def create_projector(model, dataset, dataset_sprites, writer, layer_name, args):
    model.eval()
    
    features, images, metadata = [], [], []
    with torch.no_grad():
        # generate feature bank from train dataset
        print("Extracting features from images")
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, 
            pin_memory=True, drop_last=False)
        for i, (data, _) in enumerate(data_loader):
            # print(data.size())
            feature = model(data.cuda(non_blocking=True)) # for every sample in the batch let predict features NxC tensor
            feature = torch.flatten(feature, start_dim=1)
            feature = F.normalize(feature, dim=1)
            features.append(feature.cpu())    # create list of features [tensor1 (NxC), tensor2 (NxC), tensorM (NxC)] where M is the number of minibatches
            # train_images.append(data.cpu())
            print("\t[{}]/[{}] batch iteration".format(i, len(data_loader)))

        features = torch.cat(features, dim=0).numpy() # concatenates all features tensors [NxC],[NxC],... to obtain a unique tensor of features 
                                                                    # for all the dataset DxC
        print("Collecting sprites")
        data_loader = torch.utils.data.DataLoader(dataset_sprites, batch_size=args.batch_size, shuffle=False, num_workers=1, 
            pin_memory=True, drop_last=False)
        for i, (data, _) in enumerate(data_loader):
            # print(data.size())
            images.append(data)
            print("\t[{}]/[{}] batch iteration".format(i, len(data_loader)))
        
        images = torch.cat(images, dim=0)  # same for images

        print("Collecting labels")
        for i in range(len(dataset)):
            metadata.append(dataset.metadata[i]['target']['target_level'])

        print("Creating projector")
        writer.add_embedding(features, metadata=metadata, label_img=images, tag=layer_name)

if __name__ == '__main__':
    main()
