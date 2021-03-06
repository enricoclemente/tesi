#!/usr/bin/env python
import argparse
import sys
import os

sys.path.insert(0, './')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from experiments_singlegpu.datasets.SocialProfilePictures import SocialProfilePictures
import torchvision.transforms as transforms
from experiments_singlegpu.datasets.utils.custom_transforms import PadToSquare

import torchvision.models as models

from SPICE.spice.config import Config
from SPICE.spice.model.feature_modules.resnet_cifar import resnet18_cifar

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Evaluation for MoCo with Linear Classifier')
parser.add_argument("--config_file", default="./experiments_config_example.py", metavar="FILE",
                    help="path to config file (same used with moco training)", type=str)
parser.add_argument('--dataset', default="cifar10", type=str,
                    help="name of the dataset, this lead to different script choices") 
parser.add_argument('--dataset_folder', metavar='DIR', default='./datasets/cifar10',
                    help='path to dataset')
parser.add_argument('--model_path', type=str, default='results/128_0.5_200_512_500_model.pth',
                    help='The pretrained model path')
parser.add_argument('--save_folder', metavar='DIR', default='./results/cifar10/moco',
                    help='path to results')
parser.add_argument('--logs_folder', metavar='DIR', default='./results/cifar10/moco/logs',
                    help='path to tensorboard logs')
parser.add_argument('--lr', '--learning-rate', default=0.015, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')                   
parser.add_argument('--batch-size', type=int, default=512, help='Number of images in each mini-batch')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')



def main():  
    args = parser.parse_args()
    print("linear classifier started with params:")
    print(args)
    cfg = Config.fromfile(args.config_file)

    # setting of save_folder
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    # setting of logs_folder
    if not os.path.exists(args.logs_folder):
        os.makedirs(args.logs_folder)

    # checking GPU and showing infos
    if not torch.cuda.is_available():
        raise NotImplementedError("You need GPU!")
    print("Use GPU: {} for training".format(torch.cuda.current_device()))

    torch.cuda.set_device(torch.cuda.current_device())

    encoder = models.resnet18()
    train_dataset = None
    test_dataset = None
    dataset_normalization = transforms.Normalize(mean=cfg.dataset.normalization.mean, std=cfg.dataset.normalization.std)
    
    if args.dataset == 'cifar10':
        # resnet18_cifar which is an implementation adapted for CIFAR10
        encoder = resnet18_cifar()

        # CIFAR10 train  dataset
        train_dataset = CIFAR10(root=args.dataset_folder, train=True, 
                        transform=transforms.Compose([transforms.ToTensor(), dataset_normalization]), download=True)
        test_dataset = CIFAR10(root=args.dataset_folder, train=False, 
                        transform=transforms.Compose([transforms.ToTensor(), dataset_normalization]), download=True)

    elif args.dataset == 'socialprofilepictures':
        # base resnet18 encoder since using images of the same size of ImageNet
        encoder = models.resnet18()

        resize_transform = [PadToSquare(),    # apply padding to make images squared
                            transforms.Resize([cfg.dataset.img_size, cfg.dataset.img_size])]
        # SPP train dataset 
        train_dataset = SocialProfilePictures(root=args.dataset_folder, split="train", 
                                    transform=transforms.Compose(
                                                resize_transform +
                                                [transforms.ToTensor(), 
                                                dataset_normalization]))

        # creating SPP datasets for knn test
        test_dataset = SocialProfilePictures(root=args.dataset_folder, split="test", 
                                    transform=transforms.Compose( 
                                                resize_transform + 
                                                [transforms.ToTensor(),
                                                dataset_normalization]))
    else:
        raise NotImplementedError("You must choose a valid dataset!")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    print(encoder)

    if os.path.isfile(args.model_path):
        print("Loading previously trained model on MoCo")
        loc = 'cuda:{}'.format(torch.cuda.current_device())
        checkpoint = torch.load(args.model_path, map_location=loc)
        state_dict = dict()
        for key in checkpoint['state_dict']:
            print(key)
            if key.startswith("encoder_q"):
                # print(key[22:])
                state_dict[key[10:]] = checkpoint['state_dict'][key]
        
        encoder.load_state_dict(state_dict, strict=False)
    else:
        print("Loading pretrained model on ImageNet")
        # raise NotImplementedError("You must use a pretrained feature model")
        encoder = models.resnet18(pretrained=True)

    # create new model with only query encoder
    model = Net(num_class=len(train_dataset.classes), feature_model=encoder).cuda()
    print(model)

    cudnn.benchmark = True

    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    best_acc = 0.0
    train_writer = SummaryWriter(args.logs_folder + "/linear_classifier_train")
    test_writer = SummaryWriter(args.logs_folder + "/linear_classifier_test")
    for epoch in range(1, args.epochs + 1):
        
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, criterion, optimizer, epoch, args.epochs)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, criterion, None, epoch, args.epochs)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        train_writer.add_scalar('Linear Classifier Training Loss/epoch loss',
                    train_loss,
                    epoch)
        train_writer.add_scalar('Linear Classifier Training Accuracy/epoch top1 accuracy',
                    train_acc_1,
                    epoch)
        train_writer.add_scalar('Linear Classifier Training Accuracy/epoch top5 accuracy',
                    train_acc_5,
                    epoch)
        test_writer.add_scalar('Linear Classifier Training Loss/epoch loss',
                    test_loss,
                    epoch)
        test_writer.add_scalar('Linear Classifier Training Accuracy/epoch top1 accuracy',
                    test_acc_1,
                    epoch)
        test_writer.add_scalar('Linear Classifier Training Accuracy/epoch top5 accuracy',
                    test_acc_5,
                    epoch)
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), args.save_folder + '/linear_classifier_model.pth')


# train or test for one epoch
def train_val(net, data_loader, criterion, train_optimizer, epoch, epochs):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, = 0.0, 0.0, 0.0, 0,
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for i, (data, target) in enumerate(data_loader):
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = criterion(out, target)

            if is_train:
                # print("?? il training")
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            print('{} Epoch: [{}][{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, i, len(data_loader), total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100

class Net(nn.Module):
    def __init__(self, num_class, feature_model):
        super(Net, self).__init__()

        # remove the fc layer from encoder
        self.encoder = torch.nn.Sequential(*(list(feature_model.children())[:-1]))

        # freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        #??new fc to be trained and used to classification
        self.fc = nn.Linear(512, num_class, bias=True)

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

if __name__ == '__main__':
    main()