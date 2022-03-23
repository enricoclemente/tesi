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
from spice.model.feature_modules.resnet_cifar import resnet18_cifar
from torchvision.models import resnet18

from torchvision.datasets import CIFAR10
from experiments_singlegpu.datasets.SocialProfilePictures import SocialProfilePictures

from experiments_singlegpu.datasets.utils.custom_transforms import PadToSquare



parser = argparse.ArgumentParser(description='Tensorboard Projector Creator')
parser.add_argument('--dataset_folder', metavar='DIR', default='./datasets/cifar10',
                    help='path to dataset')
parser.add_argument('--pretrained_model', default=None, type=str, metavar='PATH',
                    help='path to previously trained model')
parser.add_argument('--logs_folder', metavar='DIR', default='./results/cifar10/moco/logs',
                    help='path to tensorboard logs')
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
# dataset
parser.add_argument('--dataset', default="cifar10", type=str,
                    help="name of the dataset, this lead to different script choices")


def main():
    args = parser.parse_args()
    print(args)

    # setting of logs_folder
    if not os.path.exists(args.logs_folder):
        os.makedirs(args.logs_folder)

    # checking GPU and showing infos
    if not torch.cuda.is_available():
        raise NotImplementedError("You need GPU!")
    print("Use GPU: {} for training".format(torch.cuda.current_device()))

    torch.cuda.set_device(torch.cuda.current_device())

    model = None

    train_dataset = None
    test_dataset = None
    train_dataset_sprites = None
    test_dataset_sprites = None
    if args.dataset == 'cifar10':
        # creating model MoCo using resnet18_cifar which is an implementation adapted for CIFAR10
        model = resnet18_cifar()

        train_dataset = CIFAR10(root=args.dataset_folder, train=True, 
                transform=transforms.Compose([transforms.ToTensor()]),  
                download=True)
        test_dataset = CIFAR10(root=args.dataset_folder, train=False, 
                transform=transforms.Compose([transforms.ToTensor()]),  
                download=True)
    elif args.dataset == 'socialprofilepictures':
        model = resnet18(pretrained=True if not args.pretrained_model else False)

        train_dataset = SocialProfilePictures(root=args.dataset_folder, split='train', 
                transform=transforms.Compose([ PadToSquare(), transforms.Resize([225, 225]), transforms.ToTensor()]))
        train_dataset_sprites = SocialProfilePictures(root=args.dataset_folder, split='train', 
                transform=transforms.Compose([ PadToSquare(), transforms.Resize([64, 64]), transforms.ToTensor()]))
        test_dataset = SocialProfilePictures(root=args.dataset_folder, split=['validation'], 
                transform=transforms.Compose([ PadToSquare(), transforms.Resize([225, 225]), transforms.ToTensor()]))
        test_dataset_sprites = SocialProfilePictures(root=args.dataset_folder, split=['validation'], 
                transform=transforms.Compose([ PadToSquare(), transforms.Resize([100, 100]), transforms.ToTensor()]))
    else:
        raise NotImplementedError("Choose a valid dataset!")

    # optionally resume from a checkpoint
    if args.pretrained_model:
        if os.path.isfile(args.pretrained_model):
            print("=> loading model parameters '{}'".format(args.pretrained_model))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(torch.cuda.current_device())
            checkpoint = torch.load(args.pretrained_model, map_location=loc)
            prev_parameteres = model.parameters()
            print(checkpoint['state_dict'])
            if 'state_dict' in checkpoint.keys(): 
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            assert prev_parameteres != model.parameters(), "Model not loaded properly!"
            # print("Resume's state_dict:")
            # for param_tensor in model.state_dict():
            #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            #     print(param_tensor, "\t", model.state_dict()[param_tensor])
        else:
            raise NotImplementedError("=> no checkpoint found at '{}'".format(args.pretrained_model))


    # remove fc from encoder in order to get features
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    print(model)
    model = model.cuda()
    cudnn.benchmark = True  # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

    # tensorboard plotter
    train_writer = SummaryWriter(args.logs_folder + "/train_projector")
    test_writer = SummaryWriter(args.logs_folder + "/test_projector")

    # create_projector(model, train_dataset, train_dataset_sprites, train_writer, 'layer_4', args)
    create_projector(model, test_dataset, test_dataset_sprites, test_writer, 'layer_4', args)

    
def create_projector(model, dataset, dataset_sprites, writer, layer_name, args):
    model.eval()
    
    features, images, metadata = [], [], []
    with torch.no_grad():
        # generate feature bank from train dataset
        print("\tExtracting features from images")
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
        print("\tCollecting sprites")
        data_loader = torch.utils.data.DataLoader(dataset_sprites, batch_size=args.batch_size, shuffle=False, num_workers=1, 
            pin_memory=True, drop_last=False)
        for i, (data, _) in enumerate(data_loader):
            # print(data.size())
            images.append(data)
            print("\t[{}]/[{}] batch iteration".format(i, len(data_loader)))
        
        images = torch.cat(images, dim=0)  # same for images

        print("\tCollecting labels")
        for i in range(len(dataset)):
            metadata.append(dataset.metadata[i]['target']['target_level'])

        print("\tCreating projector")
        writer.add_embedding(features, metadata=metadata, label_img=images, tag=layer_name)

if __name__ == '__main__':
    main()
