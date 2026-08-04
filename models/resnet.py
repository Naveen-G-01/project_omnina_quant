# models/resnet.py
#
# Project Omnia addendum 
#
# A from-scratch (not torchvision-derived) ResNet-18/50 that follows the
# same `conv_layer` swap pattern as models/mobilenet.py and
# models/mobilenetv2.py: pass conv_layer=QConv2d to get a quantizable
# variant whose QConv2d/QLinear submodules expose the .w_bit/.a_bit
# attributes that entropy_quantize.py's apply_strategy() writes to.

import math
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from lib.utils.quantize_utils import QConv2d, QLinear

__all__ = ['ResNet', 'resnet18', 'resnet50', 'qresnet18', 'qresnet50']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
}

def conv3x3(in_planes, out_planes, stride=1, conv_layer=nn.Conv2d, half_wave=True):
    if conv_layer == nn.Conv2d:
        return conv_layer(in_planes, out_planes, kernel_size=3, stride=stride,
                          padding=1, bias=False)
    return conv_layer(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=1, bias=False, half_wave=half_wave)


def conv1x1(in_planes, out_planes, stride=1, conv_layer=nn.Conv2d, half_wave=True):
    if conv_layer == nn.Conv2d:
        return conv_layer(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return conv_layer(in_planes, out_planes, kernel_size=1, stride=stride,
                      bias=False, half_wave=half_wave)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, conv_layer=nn.Conv2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, conv_layer=conv_layer)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, conv_layer=conv_layer)
        self.bn2 = nn.BatchNorm2d(planes)
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

        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, conv_layer=nn.Conv2d):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, conv_layer=conv_layer)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, conv_layer=conv_layer)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, conv_layer=conv_layer)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, conv_layer=nn.Conv2d):
        super(ResNet, self).__init__()
        self.inplanes = 64

        if conv_layer == nn.Conv2d:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False, half_wave=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], conv_layer=conv_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, conv_layer=conv_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, conv_layer=conv_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, conv_layer=conv_layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if conv_layer == nn.Conv2d:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = QLinear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1, conv_layer=nn.Conv2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, conv_layer=conv_layer),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, conv_layer=conv_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_layer=conv_layer))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or type(m) == QConv2d:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) or type(m) == QLinear:
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

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
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet18'], progress=True)
        
        # If kwargs changes num_classes, strip the fc layer from the loaded weights
        if 'num_classes' in kwargs and kwargs['num_classes'] != 1000:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        
        # If kwargs changes num_classes, strip the fc layer from the loaded weights
        if 'num_classes' in kwargs and kwargs['num_classes'] != 1000:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
    return model


def qresnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], conv_layer=QConv2d, **kwargs)
    if pretrained:
        raise NotImplementedError(
            'Quantized variants load weights via entropy_quantize.py --resume '
            '<FP32 checkpoint>, not via pretrained=True.')
    return model


def qresnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], conv_layer=QConv2d, **kwargs)
    if pretrained:
        raise NotImplementedError(
            'Quantized variants load weights via entropy_quantize.py --resume '
            '<FP32 checkpoint>, not via pretrained=True.')
    return model
