#!/usr/bin/env python
import argparse
import sys
import os
sys.path.insert(0, './')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
# from thop import profile, clever_format
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from tqdm import tqdm


from spice.model.feature_modules.resnet_cifar import resnet18_cifar
import moco.builder
import moco.loader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Linear Evaluation for MoCo')
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
parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')



class Net(nn.Module):
    def __init__(self, num_class, feature_model):
        super(Net, self).__init__()

        # remove the fc layer from encoder
        self.encoder = torch.nn.Sequential(*(list(feature_model.children())[:-1]))

        # freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # new fc to be trained and used to classification
        self.fc = nn.Linear(512, num_class, bias=True)

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


# train or test for one epoch
def train_val(net, data_loader, criterion, train_optimizer, epoch, epochs):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = criterion(out, target)

            if is_train:
                # print("è il training")
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100



def main():  
    args = parser.parse_args()
    print(args)

    # Data loading code
    CIFAR10_normalization = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

    train_data = CIFAR10(root=args.dataset_folder, train=True, 
                        transform=transforms.Compose([transforms.ToTensor(),CIFAR10_normalization]), download=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_data = CIFAR10(root=args.dataset_folder, train=False, 
                        transform=transforms.Compose([transforms.ToTensor(),CIFAR10_normalization]), download=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    encoder = resnet18_cifar()

    if os.path.isfile(args.model_path):
        loc = 'cuda:{}'.format(torch.cuda.current_device())
        checkpoint = torch.load(args.model_path, map_location=loc)
        state_dict = dict()
        for key in checkpoint:
            if key.startswith("module.feature_module"):
                # print(key[22:])
                state_dict[key[22:]] = checkpoint[key]
        
        encoder.load_state_dict(state_dict, strict=False)
    else:
        raise NotImplementedError("You must use a pretrained feature model")

    # create new model with only query encoder
    model = Net(num_class=len(train_data.classes), feature_model=encoder).cuda()
    print(model)

    cudnn.benchmark = True

    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    best_acc = 0.0
    train_writer = SummaryWriter(args.logs_folder + "/train_linear_classifier")
    test_writer = SummaryWriter(args.logs_folder + "/test_linear_classifier")
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

if __name__ == '__main__':
    main()