#!/usr/bin/env python
import sys
import os
from click import version_option
sys.path.insert(0, './')

import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from SPICE.spice.data.augment import Augment, Cutout
from experiments_singlegpu.datasets.CIFAR10_custom import CIFAR10Triplet
from SPICE.spice.model.feature_modules.resnet_cifar import resnet18_cifar

import SPICE.moco.loader
import SPICE.moco.builder
import math
from SPICE.spice.config import Config
from torchvision.datasets import CIFAR10
from SPICE.spice.utils.evaluation import calculate_acc, calculate_nmi, calculate_ari
import scipy.io
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import linear_sum_assignment
from torchvision.models.resnet import resnet18
from experiments_singlegpu.datasets.utils.custom_transforms import PadToSquare
from experiments_singlegpu.datasets.SocialProfilePictures import SocialProfilePictures, SocialProfilePicturesPair, SocialProfilePicturesPro
from experiments_singlegpu.datasets.SUN397_custom import SUN397_v2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from experiments_singlegpu.datasets.utils.analysis import calculate_EMOTIC_people_perc_from_SPP
from experiments_singlegpu.datasets.utils.analysis import calculate_scenes_people_perc_from_SPP
from experiments_singlegpu.datasets.utils.analysis import calculate_SUN397_people_perc_from_SPP
from experiments_singlegpu.datasets.utils.analysis import calculate_scenes_false_positives_for_hierarchy_classes, calculate_scenes_false_positives_for_hierarchy_classes_v3
from experiments_singlegpu.datasets.utils.analysis import calculate_people_false_positives_for_hierarchy_classes




def main():  
    print("Ciao")
    calculate_scenes_false_positives_for_hierarchy_classes_v3(dataset_folder='/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets/EMOTIC/data',
                                                save_folder='/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/analysis_nonselfie_vs_scenes/',
                                                wrong_predictions_folder='/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/version_03/lda/resnet18_pretrained/exp1/train_false_positives')


def test_SPP_randomize():
    dataset = SocialProfilePictures(version=3, root='/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets',
                                    split='val', randomize_metadata=False)
    print("Dataset is big: {}".format(len(dataset)))
    print(dataset.classes_count)
    selfie = 0
    nonselfie = 0
    for img in dataset.metadata:
        if img['target']['level2'] == 'selfie':
            selfie += 1
        elif img['target']['level2'] == 'nonselfie':
            nonselfie += 1
    print(selfie)
    print(nonselfie)

def test_SPP_v2():
    dataset = SocialProfilePictures(version=3, root='/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets')
    print()
    print("Dataset is big: {}".format(len(dataset)))
    print(dataset.classes_count)
    print(dataset.metadata)

def test_SUN397_v2():
    dataset = SUN397_v2(root='/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets/SUN397/data',
                        images_people_perc_metadata='/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/version_02/analysis_people_vs_scenes/sun397/images_with_people_perc_SUN397.json', 
                        split=['train'], distribute_images=True, distribute_level='level2', partition_perc=0.2)
    print("Dataset is big: {}".format(len(dataset)))
    print(dataset.classes_count)

    
def make_analysis_people_vs_scene():
    calculate_EMOTIC_people_perc_from_SPP(dataset_folder='/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets/EMOTIC/data', 
                                        save_folder='/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/version_02/analysis_nonselfie_vs_scenes/emotic',
                                        wrong_predictions_file = '/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/version_02/moco/resnet18_pretrained/exp1/linear_classifier/train_false_positives/nonselfie_false_positives.txt',
                                        use_yolo=True, 
                                        adjust_ground_truth=False)
    calculate_scenes_people_perc_from_SPP(spp_folder='/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets',
                                        sun_folder='/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets/SUN397/data',
                                        save_folder='/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/analysis_nonselfie_vs_scenes/sun397',
                                        wrong_predictions_folder='/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/moco/resnet18_pretrained/exp1/linear_classifier/train_false_positives')
    calculate_scenes_false_positives_for_hierarchy_classes(dataset_folder='/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets/EMOTIC/data',
                                                save_folder='/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/analysis_nonselfie_vs_scenes/',
                                                wrong_predictions_folder='/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/moco/resnet18_pretrained/exp1/linear_classifier/train_false_positives')
    calculate_people_false_positives_for_hierarchy_classes(dataset_folder='/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets/EMOTIC/data',
                                                    save_folder='/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/analysis_nonselfie_vs_scenes/',
                                                    wrong_predictions_folder='/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/moco/resnet18_pretrained/exp1/linear_classifier/train_false_positives')
    calculate_SUN397_people_perc_from_SPP(dataset_folder='/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets/SUN397/data',
                                        save_folder='/scratch/work/Tesi/LucaPiano/spice/results/socialprofilepictures/version_02/analysis_nonselfie_vs_scenes/sun397')


def delete_files_by_extension(folder="./", subfolder="", extension=".txt"):
   
    files_in_directory = os.listdir(os.path.join(folder, subfolder))

    for file in files_in_directory:
        if os.path.isfile(os.path.join(folder, subfolder, file)):
           
            if file.endswith(extension):
                # print(os.path.join(folder, subfolder, file))
                print("Removing file: {}".format(os.path.join(folder, subfolder, file)))
                os.remove(os.path.join(folder, subfolder, file))
            
        else:
            delete_files_by_extension(folder, os.path.join(subfolder, file))
    
    return


def prova_new_spp():
    test_dataset = SocialProfilePicturesPro(root='/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets', split="test", 
                                    transform=transforms.Compose([
                                                transforms.Resize([224, 224]), 
                                                transforms.ToTensor(),]))

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=1, pin_memory=True)
    error_images_per_class = [ [] for c in test_dataset.classes]
    print(error_images_per_class)
    for img, target, idx in test_loader:
        print(idx)
        print(idx[0].item())
        exit()


def test_multiple_bar_chart_function():
    classes_map = {'selfie': 0, 'nonselfie': 1, 'cultural_or_historical_building_place': 2, 'transportation': 3, 'commercial_buildings': 4, 'sports_and_leisure': 5, 'sportsfields_parks_leisure_spaces': 6, 'workplace': 7, 'cultural': 8, 'man-made_elements': 9, 'home_or_hotel': 10, 'mountains_hills_desert_sky': 11, 'shopping_and_dining': 12, 'houses_cabins_gardens_and_farms': 13, 'forest_field_jungle': 14, 'water_ice_snow': 15, 'industrial_and_construction': 16, 'cat': 17, 'dog': 18, 'cartoon': 19, 'drawings': 20, 'engraving': 21, 'iconography': 22, 'painting': 23, 'sculpture': 24}

    precision_per_class, recall_per_class, f1_score_per_class = torch.empty(25, dtype=torch.long).random_(10), torch.empty(25, dtype=torch.long).random_(10), torch.empty(25, dtype=torch.long).random_(10)

    plot_multiple_bar_chart(list(classes_map.keys()), "pippo", "scores", ".", [precision_per_class, recall_per_class, f1_score_per_class], ["precision", "recall", "f1 score"])


def plot_multiple_bar_chart(x_labels, title, ylabel, save_folder, y_values, y_labels):
    
    x_values = np.array(range(0,len(x_labels)))

    width = 1.0 / len(y_values) - 0.1 * 1.0 / len(y_values)
    # create array with offsets for every bar
    widths = np.linspace(- width * (len(y_values)-1)/2, width * (len(y_values)-1)/2, len(y_values))

    plt.figure("scores", figsize=(15, 6))

    plt.gca().set(title=title, ylabel=ylabel)
    for i,y in enumerate(y_values):
        rects = plt.bar(x_values + widths[i], y, width, label=y_labels[i])
        plt.bar_label(rects, fmt='%.2f', padding=3)
    
    plt.xticks(x_values, x_labels, rotation=45, ha='right')
    plt.legend()

    # bottom, top = plt.ylim()
    # plt.ylim([bottom, top + top*0.2])
    plt.tight_layout()
    plt.savefig("{}/{}.svg".format(save_folder, title.lower().replace(" ", "_")))
    plt.close()


def prova_multiple_bar():
    classes_map = {'selfie': 0, 'nonselfie': 1, 'cultural_or_historical_building_place': 2, 'transportation': 3, 'commercial_buildings': 4, 'sports_and_leisure': 5, 'sportsfields_parks_leisure_spaces': 6, 'workplace': 7, 'cultural': 8, 'man-made_elements': 9, 'home_or_hotel': 10, 'mountains_hills_desert_sky': 11, 'shopping_and_dining': 12, 'houses_cabins_gardens_and_farms': 13, 'forest_field_jungle': 14, 'water_ice_snow': 15, 'industrial_and_construction': 16, 'cat': 17, 'dog': 18, 'cartoon': 19, 'drawings': 20, 'engraving': 21, 'iconography': 22, 'painting': 23, 'sculpture': 24}

    precision_per_class, recall_per_class, f1_score_per_class = torch.empty(25, dtype=torch.long).random_(10), torch.empty(25, dtype=torch.long).random_(10), torch.empty(25, dtype=torch.long).random_(10)

    x = list(classes_map.keys())
    print(np.array(range(0,len(x))))
    width = len(x) / (len(x) * 3) - 0.1 * len(x) / (len(x) * 3)
    plt.figure("scores", figsize=(20, 8))
    
    plt.gca().set(title="ciccio", ylabel="scores")
    rects1 = plt.bar(np.array(list(classes_map.values())) - width, precision_per_class, width, label='precision')
    rects2 = plt.bar(np.array(list(classes_map.values())), recall_per_class, width, label='recall')
    rects3 = plt.bar(np.array(list(classes_map.values())) + width, f1_score_per_class, width, label='f1 score')
    
    
    plt.xticks(np.array(list(classes_map.values())), list(classes_map.keys()), rotation=45, ha='right')
    plt.legend()

    plt.bar_label(rects1, padding=3)
    plt.bar_label(rects2, padding=3)
    plt.bar_label(rects3, padding=3)

    bottom, top = plt.ylim()
    plt.ylim([bottom, top + top*0.2])
    plt.savefig("franco.svg")
    plt.close()


def prova_error_per_class():
    map = ["franco", "mimmo", "ciccio", "piero"]
    pred = np.array([0, 1, 2, 3, 2])

    y_true = np.array([0, 3, 3, 3, 1])

    correct = pred == y_true
    print(correct)

    wrong_per_class = [ [ 0 for y in range( 4 ) ] for x in range( 4 ) ]
    print(wrong_per_class)
    for i, c in enumerate(correct):
        if c == False:
            wrong_per_class[y_true[i]][pred[i]] += 1
    print(wrong_per_class)


def test_f1_scores_2():
    scores = np.array([0.0005368671423717294, 0.0004946895733092816, 0.008510638297872339, 0.008146067415730335, 0.011258278145695364, 0.0009900990099009903, 0.0038288288288288288, 0.001680672268907563, 0.002586206896551724, 0.010344827586206895, 0.00966183574879227, 0.010027472527472528, 0.00698529411764706, 0.004166666666666667, 0.0038043478260869562, 0.009034653465346536, 0.005756578947368421, 0.004750000000000001, 0.004731457800511509, 0.002409560723514212, 0.0076347305389221545, 0.008717105263157894, 0.01178343949044586, 0.009895833333333335, 0.0096875])
    print(scores)
    scores = normalize(scores[:,np.newaxis], axis=0).ravel()
    print(scores)
    print(scores[0])

    print(max(scores))


def test_f1_score():
    test_y_pred = np.array([1, 2, 3, 4])
    print(test_y_pred)
    test_targets = np.array([0, 0, 0, 4])
    print(test_targets)
    print(f1_score(test_targets, test_y_pred, average='micro'))


def test_accuracy_calculus():
    test_y_pred = np.array([1, 2, 3, 4])
    print(test_y_pred)
    test_targets = np.array([0, 0, 0, 4])
    print(test_targets)
    acc = (test_y_pred == test_targets).sum().item() / len(test_targets) * 100
    print(acc)


def test_lda():
    loss = nn.CrossEntropyLoss()
    input = torch.randn(250, 512)
    print(input)
    target = torch.empty(250, dtype=torch.long).random_(25)
    print(target)
    clf = LinearDiscriminantAnalysis()
    franco = clf.fit(input.cpu().numpy(), target.cpu().numpy())
    prova = torch.randn(250, 512)
    print(franco.predict(prova.cpu().numpy()))
    print(franco.predict_proba(prova.cpu().numpy()).shape)
    print(franco.predict_proba(prova.cpu().numpy()))
    # print(torch.randn(26, 25))
    output = loss(torch.tensor(franco.predict_proba(input.cpu().numpy())), target)
    print(output)


def prova_interation_on_features():
    features = np.random.randint(5, size=(86, 5))
    print(features)
    batch_size = 10
    n_batches = math.ceil(len(features)/batch_size)
    for i in range(n_batches):
        start_batch = i*batch_size
        end_batch = min((i+1)*batch_size, len(features))
        print(start_batch)
        print(end_batch)
        print(features[start_batch:end_batch])


def check_resenet_flattening():
    model = resnet18()
    model = torch.nn.Sequential(*(list(model.children())[:-1])).cuda()
    print(model)
    tens = torch.randn(1,3,224,224).cuda()

    out = model(tens)
    print(out.shape)
    print(out)

    final1 = nn.Linear(512, 10, bias=True).cuda()(out)
    print(final1)
    out = torch.flatten(out, 1).cuda()
    print(out)
    final2 = nn.Linear(512, 10, bias=True).cuda()(out)
    print(final2)


def SPP_augmentations():
    dataset_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    # https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/data_aug/contrastive_learning_dataset.py#L8
    mocov2_augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([SPICE.moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        dataset_normalization
        ]
    resize_transform = [# PadToSquare(),    # apply padding to make images squared
                            transforms.Resize([224, 224])]
    # creating SPP train dataset 
    # which gave pair augmentation of image
    pair_train_dataset = SocialProfilePicturesPair(root="/scratch/work/Tesi/LucaPiano/spice/code/experiments_singlegpu/datasets", 
                            split="train", 
                                transform=transforms.Compose(
                                            resize_transform +
                                            
                                            mocov2_augmentation))
    
    pair_train_loader = torch.utils.data.DataLoader(
        pair_train_dataset, batch_size=128, shuffle=True, num_workers=1, 
        pin_memory=True, drop_last=True)
    for i, (img1, img2) in enumerate(pair_train_loader):
        if i < 3:
            plt.imshow(img1[0].numpy().transpose([1, 2, 0]) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])
            plt.savefig("aug1_{}.png".format(i))
            plt.imshow(img2[0].numpy().transpose([1, 2, 0]) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])
            plt.savefig("aug2_{}.png".format(i))
        else: 
            exit()
        

def loading_resnet_pretrained_for_moco():
    model = resnet18(pretrained=True)
    state_dict = {}
    # for name, param in model.named_parameters():
    #     print("named parameters")
    #     # if name.startswith('fc'):
    #     #     print(name)
    #     #     print(param)
    #     print(name)
    #     state_dict[name] = param
    
    for key in model.state_dict():
        if not key.startswith('fc'):
            state_dict[key] = model.state_dict()[key]
        
    modelo = resnet18(num_classes=128)
    print(modelo.state_dict()['fc.weight'])

    modelo.load_state_dict(state_dict, strict=False)
    print(modelo.state_dict()['fc.weight'])

    dim_mlp = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)


def clustering_accuracy_studies():
    y_pred = np.array([0,0,1,0,1])
    y_true = np.array([0,0,0,1,1])
    clustering_accuracy_spice(y_pred, y_true)
    clustering_accuracy_n2d(y_pred, y_true)
    # y_pred = np.array([1,1,1,0,0])
    # y_true = np.array([0,0,0,1,1])
    # spice = 1.0
    # n2d = 1.0
    # y_pred = np.array([0,0,0,1,1])
    # y_true = np.array([0,0,0,1,1])
    # spice = 1.0
    # n2d = 0.0
    # y_pred = np.array([1,1,0,0,0])
    # y_true = np.array([0,0,0,1,1])
    # spice = 0.8
    # n2d = 0.8
    # y_pred = np.array([0,0,1,1,1])
    # y_true = np.array([0,0,0,1,1])
    # spice = 0.8
    # n2d = 0.0
    # y_pred = np.array([1,1,0,1,0])
    # y_true = np.array([0,0,0,1,1])
    # spice = 0.6
    # n2d = 0.6
    # y_pred = np.array([0,0,1,0,1])
    # y_true = np.array([0,0,0,1,1])
    # spice = 0.6
    # n2d = 0.4


def clustering_accuracy_spice(y_pred, y_true):
    print("Cluster Accuracy SPICE")
    # cluster assignment
    # y_pred = np.array([0,0,1,1,1])
    print("Cluster prediction: {}".format(y_pred))
    # class labels
    # y_true = np.array([1,1,0,0,0])
    print("Ground truth: {}".format(y_true))
    # acc = calculate_acc(y_pred, y_true)
    s = np.unique(y_pred)
    t = np.unique(y_true)

    N = len(np.unique(y_pred))
    C = np.zeros((N, N), dtype=np.int32)

    for i in range(N):
        for j in range(N):
            idx = np.logical_and(y_pred == s[i], y_true == t[j])
            C[i][j] = np.count_nonzero(idx)
    Cmax = np.amax(C)
    C = Cmax - C
    print("Cost Matrix: \n{}".format(C))

    """
        Return an array of row indices and one of corresponding column indices giving the optimal assignment. 
        The cost of the assignment can be computed as cost_matrix[row_ind, col_ind].sum(). 
    """
    row, col = linear_sum_assignment(C)
    print("Linear sum assignment results")
    print("Cluster label: {}".format(row))
    print("Relative best ground truth label: {}".format(col))

    # print(C[row,col].sum())
    
    # calcolo il cluster migliore
    count = 0
    for i in range(N):
        idx = np.logical_and(y_pred == s[row[i]], y_true == t[col[i]])
        count += np.count_nonzero(idx) 
    # print(count)
    
    acc = 1.0 * count / len(y_true)
    print("Accuracy: {}".format(acc))

    # mat = linear_sum_assignment(C)
    # print(mat)

    # C = np.zeros((N, N), dtype=np.int32)

    # for i in range(N):
    #     for j in range(N):
    #         idx = np.logical_and(y_pred == s[i], y_true == t[j])
    #         C[i][j] = np.count_nonzero(idx)
    
    # acc = sum([C[i, j] for i, j in mat]) * 1.0 / y_pred.size
    # print(acc)

def clustering_accuracy_n2d(y_pred, y_true):
    print("Cluster Accuracy N2D")
    # cluster assignment
    # y_pred = np.array([1,1,0,0,0])
    print("Cluster prediction: {}".format(y_pred))
    # class labels
    # y_true = np.array([0,0,0,1,1])
    print("Ground truth: {}".format(y_true))
    y_true = y_true.astype(np.int64)

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    # print(w)

    # per ogni cluster vedo come sono suddivisi i campioni in base alle label
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # print(w)
    print("Cost Matrix: \n{}".format(w.max() - w))
    # decido a qual è per ogni label il cluster migliore (al minor costo)
    ind = linear_sum_assignment(w.max() - w)
    print("Linear sum assignment results")
    print("Cluster label: {}".format(ind[0]))
    print("Relative best ground truth label: {}".format(ind[1]))

    # print(best_fit)
    # w = w.max() - w
    acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    print("Accuracy: {}".format(acc))
    


def test_cifar10_target():
    original_train_dataset = CIFAR10(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/CIFAR10/data', train=True, 
            transform=transforms.Compose([transforms.ToTensor()]),  
            download=True)
    train_loader = torch.utils.data.DataLoader(
        original_train_dataset, batch_size=20, shuffle=True, num_workers=1, 
        pin_memory=True, drop_last=True)
    
    print(train_loader.dataset.classes)
    # print(train_loader.dataset.targets)
    for i, (img, target) in enumerate(train_loader):
        if i == 0:
            print(target)   # restituisce dei numeri relativi alla classe
            exit()


def test_new_datasets():
    mat = scipy.io.loadmat('/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/EMOTIC/data/CVPR17_Annotations.mat')
    print(mat)


def test_paper_model():
    config_file = "/scratch/work/Tesi/LucaPiano/spice/code/SPICE/configs/cifar10/spice_self.py"
    cfg = Config.fromfile(config_file)
    resume = "/scratch/work/Tesi/LucaPiano/spice/results/exp_prova/checkpoints/checkpoint_last.pth.tar"
    print("=> loading checkpoint '{}'".format(resume))
    # Map model to be loaded to specified single gpu.
    loc = 'cuda:{}'.format(torch.cuda.current_device())
    checkpoint = torch.load(resume, map_location=loc)
    state_dict = dict()
    for key in checkpoint:
        print(key)
        state_dict[key[7:]] = checkpoint[key]

        if key =='module.head.head_0.classifier.lin1.weight':
            for h in range(10):
                state_dict['head.head_{}.classifier.lin1.weight'.format(h)] = checkpoint[key]
        if key =='module.head.head_0.classifier.lin1.bias':
            for h in range(10):
                state_dict['head.head_{}.classifier.lin1.bias'.format(h)] = checkpoint[key]
        if key =='module.head.head_0.classifier.lin2.weight':
            for h in range(10):
                state_dict['head.head_{}.classifier.lin2.weight'.format(h)] = checkpoint[key]
        if key =='module.head.head_0.classifier.lin2.bias':
            for h in range(10):
                state_dict['head.head_{}.classifier.lin2.bias'.format(h)] = checkpoint[key]

    model = Sim2Sem(**cfg.model)
    feat = model.feature_module
    print("Named parameters")
    for name, param in feat.named_parameters():
        print(name)
        print(param)
    
    print("Parameters")
    for param in feat.parameters():
        print(param)

    model.cuda()
    model.load_state_dict(state_dict)
    CIFAR10_normalization = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    test_dataset = CIFAR10(root="/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/CIFAR10/data", train=False, 
        transform=transforms.Compose([transforms.ToTensor(),
                                    CIFAR10_normalization]), 
        download=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size_test, shuffle=False, 
        num_workers=1, pin_memory=True)
    
    test_writer = SummaryWriter("/scratch/work/Tesi/LucaPiano/spice/results/exp_prova/logs/test")

    print("Starting evaluation")
    best_acc = -2
    best_nmi = -1
    best_ari = -1
    best_head = -1
    best_epoch = -1
    min_loss = 1e10
    loss_head = -1
    loss_acc = -2
    loss_nmi = -1
    loss_ari = -1
    loss_epoch = -1
    for epoch in range(10):
        model.eval()

        loss_fn = nn.CrossEntropyLoss()
        num_heads = cfg.num_head
        gt_labels = []
        pred_labels = []
        scores_all = []
        accs = []
        aris = []
        nmis = []
        features_all = []
        for h in range(num_heads):
            pred_labels.append([])
            scores_all.append([])

        # extract features and cluster predictions from test dataset
        for _, (images, labels) in enumerate(test_loader):
            images = images.cuda(non_blocking=True)

            with torch.no_grad():
                pool = nn.AdaptiveAvgPool2d(1)
                features = model(images, forward_type="feature_only")
                if len(features.shape) == 4:
                    features = pool(features)
                    features = torch.flatten(features, start_dim=1)
                features = nn.functional.normalize(features, dim=1)

                scores = model(images, forward_type="sem")

            features_all.append(features)

            assert len(scores) == num_heads
            for h in range(num_heads):
                # for every image take indices of the best cluster predicted
                pred_idx = scores[h].argmax(dim=1)
                pred_labels[h].append(pred_idx)
                scores_all[h].append(scores[h])

            gt_labels.append(labels)

        gt_labels = torch.cat(gt_labels).long().cpu().numpy()
        features_all = torch.cat(features_all, dim=0)
        features_all = features_all.cuda(non_blocking=True)
        losses = []

        for h in range(num_heads):
            scores_all[h] = torch.cat(scores_all[h], dim=0)
            pred_labels[h] = torch.cat(pred_labels[h], dim=0)

        # with torch.no_grad():
        prototype_samples_indices, gt_cluster_labels = model(feas_sim=features_all, scores=scores_all, epoch=epoch,
                                                  forward_type="sim2sem")

        for h in range(num_heads):
            pred_labels_h = pred_labels[h].long().cpu().numpy()

            pred_scores_select = scores_all[h][prototype_samples_indices[h].cpu()]
            gt_labels_select = gt_cluster_labels[h]
            loss = loss_fn(pred_scores_select.cpu(), gt_labels_select)

            try:
                acc = calculate_acc(pred_labels_h, gt_labels)
            except:
                acc = -1

            nmi = calculate_nmi(pred_labels_h, gt_labels)

            ari = calculate_ari(pred_labels_h, gt_labels)

            accs.append(acc)
            nmis.append(nmi)
            aris.append(ari)

            losses.append(loss.item())

        accs = np.array(accs)
        nmis = np.array(nmis)
        aris = np.array(aris)
        losses = np.array(losses)

        # plotting results for every clu head and avgs
        loss_avg = 0
        acc_avg = 0
        ari_avg = 0
        nmi_avg = 0
        for h in range(num_heads):
            test_writer.add_scalar('Cluster Loss/epoch loss head_{}'.format(h),
                losses[h],
                epoch)
            test_writer.add_scalar('Cluster ACC/epoch acc head_{}'.format(h),
                accs[h],
                epoch)
            test_writer.add_scalar('Cluster ARI/epoch ari head_{}'.format(h),
                aris[h],
                epoch)
            test_writer.add_scalar('Cluster NMI/epoch nmi head_{}'.format(h),
                nmis[h],
                epoch)    
            loss_avg += losses[h]
            acc_avg += accs[h]
            ari_avg += aris[h]
            nmi_avg += nmis[h]
        
        loss_avg= loss_avg / num_heads
        acc_avg = acc_avg / num_heads
        ari_avg = ari_avg / num_heads
        nmi_avg = nmi_avg / num_heads
        test_writer.add_scalar('Cluster Loss/epoch loss avg',
            loss_avg,
            epoch)
        test_writer.add_scalar('Cluster ACC/epoch acc avg',
            acc_avg,
            epoch)
        test_writer.add_scalar('Cluster ARI/epoch ari avg',
            ari_avg,
            epoch)
        test_writer.add_scalar('Cluster NMI/epoch nmi avg',
            nmi_avg,
            epoch)

        best_acc_real = accs.max()
        head_real = np.where(accs == best_acc_real) # return array of indices of elements that satisfy the condition
        head_real = head_real[0][0]     # select the index value
        best_nmi_real = nmis[head_real]
        best_ari_real = aris[head_real]
        print("Real: ACC: {}, NMI: {}, ARI: {}, head: {}".format(best_acc_real, best_nmi_real, best_ari_real, head_real))
            
        test_writer.add_scalar('Cluster ACC/best acc head',
            best_acc_real,
            epoch)
        test_writer.add_scalar('Cluster NMI/best acc head',
            best_nmi_real,
            epoch)
        test_writer.add_scalar('Cluster ARI/best acc head',
            best_ari_real,
            epoch)
        test_writer.add_scalar('Cluster Loss/best acc head',
            losses[head_real],
            epoch)
        test_writer.add_scalar('Cluster Head/best acc head',
            head_real,
            epoch)

        head_loss = np.where(losses == losses.min())[0]
        head_loss = head_loss[0]
        best_acc_loss = accs[head_loss]
        best_nmi_loss = nmis[head_loss]
        best_ari_loss = aris[head_loss]
        print("Loss: ACC: {}, NMI: {}, ARI: {}, head: {}".format(best_acc_loss, best_nmi_loss, best_ari_loss, head_loss))
        test_writer.add_scalar('Cluster ACC/best loss head',
            best_acc_loss,
            epoch)
        test_writer.add_scalar('Cluster NMI/best loss head',
            best_nmi_loss,
            epoch)
        test_writer.add_scalar('Cluster ARI/best loss head',
            best_ari_loss,
            epoch)
        test_writer.add_scalar('Cluster Loss/best loss head',
            losses[head_loss],
            epoch)
        test_writer.add_scalar('Cluster Head/best loss head',
            head_loss,
            epoch)

        print("FINAL -- Best ACC: {}, Best NMI: {}, Best ARI: {}, epoch: {}, head: {}".format(best_acc, best_nmi, best_ari, best_epoch, best_head))
        print("FINAL -- Select ACC: {}, Select NMI: {}, Select ARI: {}, epoch: {}, head: {}".format(loss_acc, loss_nmi, loss_ari, loss_epoch, loss_head))



def load_paper_model():
    print("=> loading checkpoint '{}'".format("/scratch/work/Tesi/LucaPiano/spice/results/exp_prova/checkpoints/checkpoint_last.pth.tar"))
    # Map model to be loaded to specified single gpu.
    loc = 'cuda:{}'.format(torch.cuda.current_device())
    checkpoint = torch.load("/scratch/work/Tesi/LucaPiano/spice/results/exp_prova/checkpoints/checkpoint_last.pth.tar", map_location=loc)
    for key in checkpoint:
        print(key)
    

def local_consistency_model_calculus():
    # imitate the localc ocnsistency function inside sem_head_multi
    torch.set_printoptions(linewidth=150)
    
    D = 20 # number of images/samples
    N = 5 # samples per batch
    C = 10 # feature dimension
    K = 6 # clusters
    H = 3 # clustering heads
    num_neighbor = 3 # number of neighbors per image
    ratio_confident = 0.99
    score_th = 0.99

    print("We have a dataset of {} images, batch dimension is {}, number of clusters is {} and number of clustering heads is {}"
        .format(N*4, N, K, H))
    
    features = torch.randn(N, C)
    print("Features are \n {}".format(features))
    scores = torch.randn(N, K)
    print("Scores are \n {}".format(scores))

    labels_pred = scores.argmax(dim=1)
    print("Cluster label predictions (labels_pred) are \n {}".format(labels_pred))

    sim_mtx = torch.einsum('nd,cd->nc', [features, features])
    print("Self similarity matrix (sim_mtx) is \n {}".format(sim_mtx))

    scores_k, idx_k = sim_mtx.topk(k=num_neighbor, dim=1)
    print("top k={} similarities (scores_k) are \n {} \n relative indices (idx_k) are \n {}".format(num_neighbor, scores_k, idx_k))

    labels_samples = torch.zeros_like(idx_k)
    for s in range(num_neighbor):
        # every column of label_samples
        labels_samples[:, s] = labels_pred[idx_k[:, s]]
    print("Label samples (labels_samples) are \n {}".format(labels_samples))

    true_mtx = labels_samples[:, 0:1] == labels_samples
    print("True matrix (true_mtx) is \n {}".format(true_mtx))

    num_true = true_mtx.sum(dim=1)
    print("Number of neighbors with same cluster (num_true) is \n {}".format(num_true))

    idx_true = num_true >= num_neighbor * ratio_confident
    print("idx_true \n {}".format(idx_true))

    idx_conf = scores.max(dim=1)[0] > score_th
    print("idx_conf \n {}".format(idx_conf))

    idx_true = idx_true * idx_conf
    print("See which samples are reliable (idx_true*idx_conf) \n {}".format(idx_true))

    idx_select = torch.where(idx_true > 0)[0]
    print("Reliable samples (idx_select) \n {}".format(idx_select))

    labels_select = labels_pred[idx_select]


def train_moco_prova_cosine_schedule():
   
    fig, ax = plt.subplots()
    lr = 0.06
    epochs = 500
    lrs = []
    for epoch in range(epochs):
        lr = 0.06
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
        lrs.append(lr)
    print("Cosine lr schedule with 500 epochs")
    print(lrs)
    ax.plot(range(epochs), lrs, linewidth=2.0)
    plt.savefig("cosine_lr_500_epochs.svg")

    fig, bx = plt.subplots()
    epochs = 700
    last_lr = lr
    for epoch in range(epochs):
        if epoch >= 500:
            lr = last_lr
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
            lrs.append(lr)
    print("Cosine lr schedule adding 200 epochs")
    print(lrs)
    bx.plot(range(epochs), lrs, linewidth=2.0)
    plt.savefig("cosine_lr_adding_200_epochs.svg")

    fig, cx = plt.subplots()
    lr = 0.06
    epochs = 700
    lrs = []
    for epoch in range(epochs):
        lr = 0.06
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
        lrs.append(lr)
    print("Cosine lr schedule with 700 epochs")
    print(lrs)
    cx.plot(range(epochs), lrs, linewidth=2.0)
    plt.savefig("cosine_lr_700_epochs.svg")



def train_spice_prova_scores():
    # simulazione di quello che accade quando si lancia la funziona select_samples del modello SPICE
    # for printing better tensors
    torch.set_printoptions(linewidth=150)
    
    D = 20 # number of images/samples
    N = 5 # samples per batch
    C = 10 # feature dimension
    K = 6 # clusters
    H = 3 # clustering heads

    

    print("We have a dataset of {} images, batch dimension is {}, number of clusters is {} and number of clustering heads is {}"
        .format(N*4, N, K, H))
    
    # features for all the dataset
    features = torch.randn(D, C)
    print("Features for the images are \n {}".format(features))

    scores = []

    # 3 clustering heads
    scores.append([])
    scores.append([])
    scores.append([])

    # dataset di 4 minbatch di 4 immagini
    for batch in range(int(D/N)):
        scores_calculated = []
        # simula calcolo del modello che ritorna lista di tensori di scores
        for head in range(H):
            scores_calculated.append(torch.randn(N,K))
        
        print("calculated scores for batch {} are: \n {}".format(batch, scores_calculated))
        # per ogni rispettiva testa viene attaccato il calcolo per gli scores di tutto il dataset
        for head in range(H):
            scores[head].append(scores_calculated[head])
        # print("total scores are: \n {}".format(scores))
    
    for head in range(H):
        scores[head] = torch.cat(scores[head], dim=0)
    
    print("total scores are: \n {}".format(scores))

    print("Getting best scores per cluster")
    center_ratio = 0.5
    samples_per_cluster = D // K
    ratio_select = 2
    k = int(center_ratio * samples_per_cluster * ratio_select)
    # for every clustering head calculate the 
    for head in range(H):
        score = scores[head]
        # order scores per cluster by ordering scores per image by column
        score_sorted, indices = torch.sort(score, dim=0, descending=True)
        print("for head {} \nsorted scores by column: \n {} \n indices of sorted scores: \n {}"
            .format(head, score_sorted, indices))
        # select the best k scores per cluster
        indices_best = indices[0:k, :]
        print("for head {} best {} scores per cluster are the images: \n {}"
            .format(head, k, indices_best))
        
        centers = []
        for cluster in range(K):
            cluster_best_samples = indices_best[:, cluster]
            print("cluster best samples for cluster {} are \n {}".format(cluster, cluster_best_samples))
            cluster_best_features = features[cluster_best_samples, :]
            print("relative features are \n {} ".format(cluster_best_features))
            center = cluster_best_features.mean(axis=0)
            print("cluster center is \n {} size {}".format(center, center.size()))
            center = center.unsqueeze(dim=0)
            print("unsqueezed cluster center is \n {} size {}".format(center, center.size()))
            centers.append(center)

        print("centers are \n {}".format(centers))

        centers = torch.cat(centers, dim=0)
        print("applying torch.cat to centers \n {}".format(centers))

        cosine_similarities = torch.einsum('kc,nc->kn', [centers, features])
        print("cosine distances between clusering centers and features are \n {}".format(cosine_similarities))

        similarities_best_indices = torch.argsort(cosine_similarities, dim=1, descending=True) 
        print("best similarities are \n {}".format(similarities_best_indices))

        similarities_best_indices = similarities_best_indices[:, 0:samples_per_cluster*ratio_select].flatten()
        print("best similarities for first {} samples per cluster are \n {}".format(samples_per_cluster*ratio_select, similarities_best_indices))
        print(similarities_best_indices.size())
        # create tensor of values [0, 1, ... K-1]
        # unsqueeze it into tensor [k,1] (column with unique values)
        # enlarge every row tensor replicating values num_select_c times --> tensor [K, num_select_c]
        # transorm again tensor into a one dimension tensor [K*num_select_c]
        labels_select = torch.arange(0, K).unsqueeze(dim=1).repeat(1, samples_per_cluster*ratio_select).flatten()
        print("labels select are \n {}".format(labels_select))
        print(labels_select.size())



def train_spice_prova_CIFARTriplet():
    CIFAR10_normalization = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    
    # weak augmentation
    weak_augmentation = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            CIFAR10_normalization
        ])

    # strong augmentation
    strong_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            Augment(4),
            transforms.ToTensor(),
            CIFAR10_normalization,
            Cutout(
                n_holes=1,
                length=16,
                random=True)])
    
    # dataset will retrieve 3 images: the original, one weakly augmented, one strong augmented
    train_dataset = CIFAR10Triplet(root='/scratch/work/Tesi/LucaPiano/spice/code/SPICE/experiments_singlegpu/datasets/CIFAR10/data', 
        train=True, 
        transform=dict(augmentation_1=weak_augmentation, augmentation_2=strong_augmentation),
        download=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=1, 
        pin_memory=True, drop_last=True)

    for i, ( img, aug1, aug2) in enumerate(train_loader):
        if i == 0:
            fig = plt.figure()
            plt.imshow(img[0].numpy().transpose([1, 2, 0]) * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465])
            plt.savefig("original_img.png")
            fig = plt.figure()
            plt.imshow(aug1[0].numpy().transpose([1, 2, 0]) * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465])
            plt.savefig("transform_img1.png")
            fig = plt.figure()
            plt.imshow(aug2[0].numpy().transpose([1, 2, 0]) * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465])
            plt.savefig("transform_img2.png")
        print("ciao")




if __name__ == '__main__':
    main()

    