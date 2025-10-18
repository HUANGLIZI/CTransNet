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
from eval import validate
from eval_os import validate_DFS_Reg
from sklearn.model_selection import KFold
from losses import Loglike_loss, L2_Regu_loss, NegativeLogLikelihood, cox_loss, neg_par_log_likelihood

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./dataset/SMU/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='multimodal_reproduce', help='experiment_name')
parser.add_argument('--num_classes', type=str,  default="three",
                    help='output channel of network')
parser.add_argument('--model', type=str,
                    default='resnet50&Transformer', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
args = parser.parse_args()


def train(args, snapshot_path):
    base_lr = args.base_lr
    if args.num_classes == "two":
        num_classes = 2
    elif args.num_classes == "three":
        num_classes = 3
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    print(num_classes)
    config_vit = config.get_CTranS_config()
    model = CTransNet(config_vit, num_classes, image_feature_length=1000, radiomics_feature_length=584,
                                 clinical_feature_length=9, feature_planes=128).cuda()
    db_train = BaseDataSet(base_dir=args.root_path, split="train", classes=args.num_classes, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSet(base_dir=args.root_path, split="val", classes=args.num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.00001)
    ce_loss = CrossEntropyLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("The training set has {} images".format(len(db_train)))
    logging.info("The validation set has {} images".format(len(db_val)))

    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_auc = 0.0
    best_acc = 0.0
    best_cindex = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, os_batch = sampled_batch['image'].cuda(
            ), sampled_batch['diagnosis'].cuda(), sampled_batch['os'].cuda()
            radimocis, clinical = sampled_batch['radimocis'].cuda(
            ), sampled_batch["clinical"].cuda()
            outputs = model(volume_batch, radimocis, clinical)
            contrastive_loss = nn.CosineEmbeddingLoss(margin=0.5)
            loss = NegativeLogLikelihood(-outputs[2], os_batch[:, 0], os_batch[:, 1])*0.5+L2_Regu_loss(weights=outputs[1], alpha=0.1)*0.5 + ce_loss(outputs[0], label_batch.long())+ contrastive_loss(outputs[3], outputs[4], torch.ones(volume_batch.shape[0]).cuda())*0.1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            logging.info(
                'iteration %d : loss : %f' %
                (iter_num, loss.item()))

            if iter_num > 0 and iter_num % 20 == 0:
                acc, sensitivity, specificity, precision,  F1, auc, kappa, id_list, results_list, results_list_pred,labels_list = validate(
                    model, valloader, num_classes)
                valid_cindex, id_list_reg, Survival_time, Survival_label = validate_DFS_Reg(
                    model, valloader, num_classes)
                
                writer.add_scalar('info/valid_cindex', valid_cindex, iter_num)
                writer.add_scalar('info/acc', acc, iter_num)
                writer.add_scalar('info/sensitivity', sensitivity, iter_num)
                writer.add_scalar('info/specificity', specificity, iter_num)
                writer.add_scalar('info/precision', precision, iter_num)
                writer.add_scalar('info/F1', F1, iter_num)
                writer.add_scalar('info/auc', auc, iter_num)
                writer.add_scalar('info/kappa', kappa, iter_num)

                if acc > best_acc or valid_cindex > best_cindex:
                    if acc > best_acc:
                        best_acc = acc
                        best_auc = auc
                        dict = {'ImageID': id_list, 'pred_prob': results_list, 'pred':results_list_pred, 'label': labels_list}
                        df = pd.DataFrame(dict)
                        df.to_csv('val_pred.csv',index=False)
                    if valid_cindex > best_cindex:
                        dict_reg  = {'ImageID': id_list_reg, 'pred': Survival_time, 'label': Survival_label}
                        df_reg = pd.DataFrame(dict_reg)
                        df_reg.to_csv('val_pred_reg.csv',index=False)
                        best_cindex = valid_cindex
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_auc_{}_acc_{}_cindex_{}.pth'.format(
                                                      iter_num, round(auc, 4), round(acc, 4), round(valid_cindex, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : auc : %f acc : %f' % (iter_num, auc, acc))
                logging.info(
                    'iteration %d : cindex : %f ' % (iter_num, valid_cindex))
                model.train()

            if iter_num % 250 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "./model/{}/{}_{}".format(
        args.exp, args.model, args.num_classes)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
    
