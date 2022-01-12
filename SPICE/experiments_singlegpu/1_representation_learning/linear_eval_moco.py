#!/usr/bin/env python
import sys
sys.path.insert(0, './')
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
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
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')


class Net(nn.Module):
    def __init__(self, num_class, moco_model, pretrained_path):
        super(Net, self).__init__()

        # take query encoder of moco
        self.encoder_q = moco_model.encoder_q
        # classifier
        # linear_in_features = self.encoder_q.fc.in_features

        for param_tensor in self.encoder_q.state_dict():
            print(param_tensor, "\t", self.encoder_q.state_dict()[param_tensor].size())
        print(self.encoder_q.state_dict()['layer4.1.bn2.weight'])

        # load trained parameters
        loc = 'cuda:{}'.format(torch.cuda.current_device())
        checkpoint = torch.load(pretrained_path, map_location=loc)
        self.load_state_dict(checkpoint['state_dict'], strict=False)
        # print(self.encoder_q.state_dict()['layer4.1.bn2.weight'])

        # remove the fc layer from encoder_q
        self.encoder_q = torch.nn.Sequential(*(list(self.encoder_q.children())[:-1]))
        # print(self.encoder_q.state_dict()['5.1.bn2.weight'])

        self.fc = nn.Linear(512, num_class, bias=True)

    def forward(self, x):
        x = self.encoder_q(x)
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
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_data = CIFAR10(root=args.dataset_folder, train=False, 
                        transform=transforms.Compose([transforms.ToTensor(),CIFAR10_normalization]), download=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    moco_model = moco.builder.MoCo(
        base_encoder=resnet18_cifar,
        dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t, mlp=args.mlp, input_size=32, single_gpu=True)

    # create new model with only query encoder
    model = Net(num_class=len(train_data.classes), moco_model=moco_model, pretrained_path=args.model_path).cuda()

    print(model)

    for param in model.encoder_q.parameters():
        param.requires_grad = False

    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    # flops, params = clever_format([flops, params])

    # print('# Model Params: {} FLOPs: {}'.format(params, flops))

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