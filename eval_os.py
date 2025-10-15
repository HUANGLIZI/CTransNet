"""Test script for ATDA."""

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import pandas
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
from sklearn.metrics import cohen_kappa_score
import torch
import torch.nn.functional as F
from lifelines.utils import concordance_index

import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import pandas as pd
import Config as config

from dataloader.dataset import BaseDataSet, RandomGenerator
from nets.resnet import resnet18, MultiModalClassifier, resnet50
from nets.CTransNet import CTransNet
from sklearn.model_selection import KFold
from losses import Loglike_loss, L2_Regu_loss, NegativeLogLikelihood, cox_loss, neg_par_log_likelihood

t = 0.5


def Get_survival_time(Survival_pred):

    breaks = np.array([0.0, 8.0, 12.0, 19.0, 33.0, 115.0])

    intervals = breaks[1:] - breaks[:-1]
    n_intervals = len(intervals)

    Survival_time = 0
    for i in range(n_intervals):
        cumulative_prob = np.prod(Survival_pred[0:i+1])
        Survival_time = Survival_time + cumulative_prob * intervals[i]

    return Survival_time


def validate_DFS(model, data_loader, num_classes):

    model.eval()
    Survival_time = []
    Survival_label = []
    # evaluate network
    with torch.no_grad():
        for sampled_batch in data_loader:
            volume_batch, label_batch = sampled_batch['image'].cuda(
            ), sampled_batch['os'].cuda()
            radimocis, clinical = sampled_batch['radimocis'].cuda(
            ), sampled_batch["clinical"].cuda()

            preds = model(volume_batch, radimocis, clinical)[-1]

            Survival_label.append(label_batch.data.cpu().numpy()[0])
            Survival_pred = preds.detach().cpu().numpy().squeeze()
            Survival_time.append(Get_survival_time(Survival_pred))

        valid_cindex = concordance_index(np.array(Survival_label)[
                                         :, 0], Survival_time, np.array(Survival_label)[:, 1])

        return valid_cindex


def validate_DFS_Reg(model, data_loader, num_classes):

    model.eval()
    Survival_time = []
    Survival_label = []
    id_list = []
    # evaluate network
    with torch.no_grad():
        for sampled_batch in data_loader:
            volume_batch, label_batch = sampled_batch['image'].cuda(
            ), sampled_batch['os'].cuda()
            radimocis, clinical = sampled_batch['radimocis'].cuda(
            ), sampled_batch["clinical"].cuda()

            preds = model(volume_batch, radimocis, clinical)[2]

            id_list.extend(sampled_batch['id'].data.cpu().numpy())
            Survival_label.append(label_batch.data.cpu().numpy()[0])
            # Survival_pred = preds.detach().cpu().numpy().squeeze()
            Survival_time.append(preds.detach().cpu().numpy().squeeze())

        valid_cindex = concordance_index(np.array(Survival_label)[
                                         :, 0], Survival_time, np.array(Survival_label)[:, 1])

        return valid_cindex, id_list, Survival_time, Survival_label

# parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str,
#                     default='../dataset/SMU/', help='Name of Experiment')
# parser.add_argument('--exp', type=str,
#                     default='multimodal_reproduce', help='experiment_name')
# parser.add_argument('--num_classes', type=str,  default="three",
#                     help='output channel of network')
# parser.add_argument('--model', type=str,
#                     default='resnet50&Transformer', help='model_name')
# parser.add_argument('--max_iterations', type=int,
#                     default=6000, help='maximum epoch number to train')
# parser.add_argument('--batch_size', type=int, default=24,
#                     help='batch_size per gpu')
# parser.add_argument('--deterministic', type=int,  default=1,
#                     help='whether use deterministic training')
# parser.add_argument('--base_lr', type=float,  default=0.0001,
#                     help='segmentation network learning rate')
# parser.add_argument('--patch_size', type=list,  default=[224, 224],
#                     help='patch size of network input')
# parser.add_argument('--seed', type=int,  default=1337, help='random seed')
# args = parser.parse_args()

# base_lr = args.base_lr
# num_classes = 3
# batch_size = 1

# config_vit = config.get_CTranS_config()
# model = CTransNet(config_vit, num_classes, image_feature_length=1000, radiomics_feature_length=584,
#                                  clinical_feature_length=9, feature_planes=128).cuda()
# model.load_state_dict(torch.load('/lizihan/lzh/SMU-GC-Cls/model/multimodal_reproduce/resnet50_Transformer_three/iter_2000.pth'))
# # db = BaseDataSet(base_dir=args.root_path, split="train", classes=args.num_classes)
# db_val = BaseDataSet(base_dir=args.root_path, split="val", classes=args.num_classes)

# valloader = DataLoader(db_val, batch_size=1, shuffle=False,
#                            num_workers=1)

# valid_cindex, id_list_reg, Survival_time, Survival_label = validate_DFS_Reg(
#                     model, valloader, num_classes)
# print(valid_cindex)
# dict_reg  = {'ImageID': id_list_reg, 'pred': Survival_time, 'label': Survival_label}
# df_reg = pd.DataFrame(dict_reg)
# df_reg.to_csv('train_pred_reg.csv',index=False)
