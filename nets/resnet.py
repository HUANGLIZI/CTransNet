import torch
import torch.nn as nn
import torch.nn.functional as F
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


class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes=2, image_feature_length=1000, radiomics_feature_length=538, clinical_feature_length=34, ihc_feature_length=20, feature_planes=256):
        super(MultiModalClassifier, self).__init__()
        self.fc1 = nn.Linear(image_feature_length, feature_planes)
        self.bn1 = nn.BatchNorm1d(feature_planes)
        self.fc2 = nn.Linear(radiomics_feature_length, feature_planes)
        self.bn2 = nn.BatchNorm1d(feature_planes)
        self.fc3 = nn.Linear(clinical_feature_length, feature_planes)
        self.bn3 = nn.BatchNorm1d(feature_planes)
        self.fc4 = nn.Linear(ihc_feature_length, feature_planes)
        self.bn4 = nn.BatchNorm1d(feature_planes)
        self.fc5 = nn.Linear(feature_planes * 4, feature_planes)
        self.bn5 = nn.BatchNorm1d(feature_planes)
        self.cls = nn.Linear(feature_planes, num_classes)

        self.drop = nn.Dropout(0.3)
        self.relu = nn.ReLU(inplace=True)
        self.feature_extractor = resnet18(pretrained=True)

    def forward(self, image, radiomics_feature, clinical_feature, ihc_feature):
        image_feature = self.feature_extractor(image)
        x1 = self.relu(self.bn1(self.fc1(self.drop(self.relu(image_feature)))))
        x1 = self.drop(x1)
        x2 = self.relu(self.bn2(self.fc2(radiomics_feature)))
        x2 = self.drop(x2)
        x3 = self.relu(self.bn3(self.fc3(clinical_feature)))
        x3 = self.drop(x3)
        x4 = self.relu(self.bn4(self.fc4(ihc_feature)))
        x4 = self.drop(x4)

        x_fusion = torch.cat((x1, x2, x3, x4), dim=1)
        x_short = self.relu(self.bn5(self.fc5(x_fusion)))
        x_short = self.drop(x_short)
        cls = self.cls(x_short)
        return cls


class MultiModalClassifier_noIHC(nn.Module):
    def __init__(self, num_classes=2, image_feature_length=1000, radiomics_feature_length=538, clinical_feature_length=34, feature_planes=256):
        super(MultiModalClassifier_noIHC, self).__init__()
        self.fc1 = nn.Linear(image_feature_length, feature_planes)
        self.bn1 = nn.BatchNorm1d(feature_planes)
        self.fc2 = nn.Linear(radiomics_feature_length, feature_planes)
        self.bn2 = nn.BatchNorm1d(feature_planes)
        self.fc3 = nn.Linear(clinical_feature_length, feature_planes)
        self.bn3 = nn.BatchNorm1d(feature_planes)

        self.fc5 = nn.Linear(feature_planes * 3, feature_planes)
        self.bn5 = nn.BatchNorm1d(feature_planes)
        self.cls = nn.Linear(feature_planes, num_classes)

        self.drop = nn.Dropout(0.3)
        self.relu = nn.ReLU(inplace=True)
        self.feature_extractor = resnet18(pretrained=True)

    def forward(self, image, radiomics_feature, clinical_feature):
        image_feature = self.feature_extractor(image)
        x1 = self.relu(self.bn1(self.fc1(self.drop(self.relu(image_feature)))))
        x1 = self.drop(x1)
        x2 = self.relu(self.bn2(self.fc2(radiomics_feature)))
        x2 = self.drop(x2)
        x3 = self.relu(self.bn3(self.fc3(clinical_feature)))
        x3 = self.drop(x3)

        x_fusion = torch.cat((x1, x2, x3), dim=1)
        x_short = self.relu(self.bn5(self.fc5(x_fusion)))
        x_short = self.drop(x_short)
        cls = self.cls(x_short)
        return cls


class Image_based_Classifier(nn.Module):
    def __init__(self, num_classes=2, image_feature_length=1000):
        super(Image_based_Classifier, self).__init__()
        self.fc1 = nn.Linear(image_feature_length, num_classes)
        self.drop = nn.Dropout(0.3)
        self.relu = nn.ReLU(inplace=True)
        self.feature_extractor = resnet18(pretrained=True)

    def forward(self, image):
        image_feature = self.feature_extractor(image)
        x = self.fc1(self.drop(self.relu(image_feature)))
        return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, layers_dic['resnet18'], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(urls_dic['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, layers_dic['resnet34'], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(urls_dic['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, layers_dic['resnet50'], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(urls_dic['resnet50']))
    return model
