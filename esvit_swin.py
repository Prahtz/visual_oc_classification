import torch
from torch import nn

import esvit.utils as utils
from esvit.models import build_model

from esvit.config import config
from esvit.config import update_config

from easydict import EasyDict


class EsVitPreTrained(nn.Module):
    def __init__(self, cfg, arch, checkpoint_path, num_blocks=4, trainable=True):
        super(EsVitPreTrained, self).__init__()
        self.arch = arch
        self.num_blocks = num_blocks
        self.depths = {'swin_tiny': [2, 2, 6, 2], 
                       'swin_base': [2, 2, 18, 2], 
                       'swin_small': [2, 2, 18, 2]}

        self.esvit = self.__get_model(cfg, arch, checkpoint_path)
        if not trainable:
            for param in self.esvit.parameters():
                param.requires_grad = False

    
    def forward(self, x):
        x = self.esvit.forward_return_n_last_blocks(x, self.num_blocks, True, depth=self.depths[self.arch])
        x = nn.functional.normalize(x, dim=1, p=2)
        return x
    
    def __get_model(self, cfg, arch, checkpoint_path):
        args = EasyDict({
            'cfg':cfg,
            'arch':arch,
            'pretrained_weights': checkpoint_path,
            'opts': ['MODEL.NUM_CLASSES', '0'],
            'rank': 0,
            'checkpoint_key': 'teacher',
            'patch_size': 16,
        })

        update_config(config, args)
        model = build_model(config, is_teacher=True)
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
        return model

def extract_features(model, data_loader):
    model.eval()
    with torch.no_grad():
        features = []
        labels = []
        for data in data_loader:
            input, label = data
            input = input.cuda()
            feats = model(input)
            features.append(feats.cpu())
            labels.append(label)
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
    return features, labels

""" def get_model(cfg, arch, checkpoint_path, checkpoint_key):
    args = EasyDict({
        'cfg':cfg,
        'arch':arch,
        'pretrained_weights': checkpoint_path,
        'opts': ['MODEL.NUM_CLASSES', '0'],
        'rank': 0,
        'checkpoint_key': checkpoint_key,
        'patch_size': 16,
    })

    update_config(config, args)
    model = build_model(config, is_teacher=True)
    return model, args

def load_weigths(model, args):
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)


def extract_features(model, data_loader, stages=[0, 1, 2, 3], normalize=True):
    model.eval()
    with torch.no_grad():
        features = [[] for i in range(4)]
        labels = [[] for i in range(4)]

        for i, data in enumerate(data_loader):
            x, label = data
            x = x.cuda()
            x = model.patch_embed(x)
            if model.ape:
                x = x + model.absolute_pos_embed
            x = model.pos_drop(x)
            for j, layer in enumerate(model.layers):
                if j in stages:
                    x, feats = layer.forward_with_features(x)
                    feats = feats[-1]
                    if j == 3: # use the norm in the last stage
                        feats = model.norm(feats)
                    feats = torch.flatten(model.avgpool(feats.transpose(1, 2)), 1)  # B C     
                    features[j].append(feats.cpu())
                    labels[j].append(label)
                else:
                    x = layer.forward(x)
        for i in stages:
            features[i] = torch.cat(features[i], dim=0)
            labels[i] = torch.cat(labels[i], dim=0)
            if normalize:
                features[i] = nn.functional.normalize(features[i], dim=1, p=2)
        return features, labels """