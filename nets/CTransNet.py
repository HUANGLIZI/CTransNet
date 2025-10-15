# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .Vit import *
from .VLAB import VLAlignBlock
from sklearn.metrics.pairwise import cosine_similarity
import torch.utils.model_zoo as model_zoo

urls_dic = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

layers_dic = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
    'resnet152': [3, 8, 36, 3]
}

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, batch_norm_fn=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = batch_norm_fn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = batch_norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.vlab = VLAlignBlock(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.vlab(skip_x)
        x = torch.cat([skip_x_att, up], dim=1)
        return self.nConvs(x),skip_x_att

class ResNet(nn.Module):

    def __init__(self, block, layers, strides=(2, 2, 2, 2), dilations=(1, 1, 1, 1), batch_norm_fn=nn.BatchNorm2d):
        self.batch_norm_fn = batch_norm_fn

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=strides[0], padding=3,
                               bias=False)
        self.bn1 = self.batch_norm_fn(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=1, dilation=dilations[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=strides[3], dilation=dilations[3])
        self.inplanes = 1024

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, 1000)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.batch_norm_fn(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample,
                        dilation=1, batch_norm_fn=self.batch_norm_fn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                          dilation=dilation, batch_norm_fn=self.batch_norm_fn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, batch_norm_fn=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = batch_norm_fn(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = batch_norm_fn(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = batch_norm_fn(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, layers_dic['resnet50'], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(urls_dic['resnet50']))
    return model

class CTransNet(nn.Module):
    def __init__(self, config, n_classes, image_feature_length=1000, radiomics_feature_length=538, clinical_feature_length=34, feature_planes=256, vis=False):
        super().__init__()
        self.vis = vis
        self.depth = 4
        self.image2vit = nn.Linear(image_feature_length, 256)
        self.radiomics2vit = nn.Linear(radiomics_feature_length, 256)
        self.clinical2vit = nn.Linear(clinical_feature_length, 256)
        self.image2vit_reg = nn.Linear(image_feature_length, 256)
        self.radiomics2vit_reg = nn.Linear(radiomics_feature_length, 256)
        self.clinical2vit_reg = nn.Linear(clinical_feature_length, 256)
        # self.ihc2vit = nn.Linear(ihc_feature_length, 256)
        self.Encoder_blocks = nn.Sequential(*[
            Block(dim=256, num_heads=8, mlp_ratio=4.0)
            for _ in range(self.depth)])
        self.Encoder_blocks_reg = nn.Sequential(*[
            Block(dim=256, num_heads=8, mlp_ratio=4.0)
            for _ in range(self.depth)])
        self.feature_extractor = resnet50(pretrained=True)
        self.feature_extractor_reg = resnet50(pretrained=True)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.3)
        self.cls_decoder = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.regress_decoder = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.cls_head = nn.Linear(64, n_classes)
        self.regress_head = nn.Linear(64, 1)

    def forward(self, image, radiomics_feature, clinical_feature):
        image_feature = self.feature_extractor(image)  # [2, 1000]
        image_feature_reg = self.feature_extractor(image)  # [2, 1000]
        image_feature_cls = self.drop(self.relu(self.image2vit(image_feature)))
        image_feature_reg = self.drop(self.relu(self.image2vit_reg(image_feature_reg)))
        image_feature_reg = image_feature_reg.unsqueeze(1)
        image_feature_cls = image_feature_cls.unsqueeze(1)

        clinical_feature_reg = self.drop(self.relu(self.clinical2vit_reg(clinical_feature)))
        clinical_feature = self.drop(self.relu(self.clinical2vit(clinical_feature)))
        clinical_feature_reg = clinical_feature_reg.unsqueeze(1)
        clinical_feature = clinical_feature.unsqueeze(1)

        # ihc_feature = self.drop(self.relu(self.ihc2vit(ihc_feature)))
        # ihc_feature = ihc_feature.unsqueeze(1)

        radiomics_feature_reg = self.drop(self.relu(self.radiomics2vit_reg(radiomics_feature)))
        radiomics_feature = self.drop(self.relu(self.radiomics2vit(radiomics_feature)))
        radiomics_feature_reg = radiomics_feature_reg.unsqueeze(1)
        radiomics_feature = radiomics_feature.unsqueeze(1)

        x = torch.cat([image_feature_cls, radiomics_feature, clinical_feature], dim=1)
        # x_reg = torch.cat([image_feature_reg, radiomics_feature_reg, clinical_feature_reg], dim=1)
        x_reg = image_feature_reg + radiomics_feature_reg + clinical_feature_reg
        x = self.drop(self.Encoder_blocks(x).transpose(1, 2).flatten(1))
        x_reg = self.drop(self.Encoder_blocks_reg(x_reg).transpose(1, 2).flatten(1))
        cls_feature = self.cls_decoder(x)
        reg_feature = self.regress_decoder(x_reg)
        pred_cls = self.cls_head(cls_feature)
        pred_reg = self.regress_head(reg_feature)
        
        return pred_cls, self.regress_head.weight, pred_reg, cls_feature, reg_feature
