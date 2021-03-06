import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn


def strip_empty_str(strings):
    while strings and strings[-1] == "":
        del strings[-1]
    return strings


def extract_features_targets(model, data_loader, cfg, normalize=False):
    model.eval()
    features = []
    targets = []
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data = data.cuda()
            
            if cfg.model == "resnet18":
                feature = model.forward(data)
                feature = torch.flatten(feature, 1)
            elif cfg.model == "vit_b_32":
                feature = model.encode_image(data)
            
            if normalize: 
                feature = F.normalize(feature, dim=1)

            #¬†collecting all features and targets
            features.append(feature.cpu())
            targets.append(target)
            
            print("\t[{}]/[{}] batch iteration".format(i, len(data_loader)))
    features = torch.cat(features).numpy()
    targets = torch.cat(targets).numpy()

    return features, targets

def extract_features_targets_indices(model, data_loader, cfg, normalize=False):
    model.eval()
    features = []
    targets = []
    indices = []
    with torch.no_grad():
        for i, (data, target, idx) in enumerate(data_loader):
            data = data.cuda()
           
            if cfg.model == "resnet18":
                feature = model.forward(data)
                feature = torch.flatten(feature, 1)
            elif cfg.model == "vit_b_32":
                feature = model.encode_image(data)

            if normalize: 
                feature = F.normalize(feature, dim=1)

            #¬†collecting all features and targets
            features.append(feature.cpu())
            targets.append(target)
            indices.append(idx)
            
            print("\t[{}]/[{}] batch iteration".format(i, len(data_loader)))
    features = torch.cat(features).numpy()
    targets = torch.cat(targets).numpy()
    indices = torch.cat(indices).numpy()

    return features, targets, indices