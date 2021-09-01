import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor


class BasicBlockTimeBest(nn.Module):  # изменен лики релу 0.1 на 0.01
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlockTimeBest, self).__init__()
        if stride == 1 or stride == (1, 1):
            kernel_size = (3, 3)
            kernel_size_2 = (3, 3)
        elif stride == (2, 2) or stride == (2, 2) or stride == (2, 1) or stride == (1, 2):
            kernel_size = (3, 3)
            kernel_size_2 = (3, 3)
        elif stride == (4, 2):
            kernel_size = (6, 3)
            kernel_size_2 = (5, 3)
        elif stride == (8, 2):
            kernel_size = (10, 3)
            kernel_size_2 = (5, 3)

        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size_2, stride=1,
                               padding=(2 if kernel_size_2[0] == 5 else 1, 1), bias=False)
        self.droprate = dropRate
        self.equalInOut =  (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
    

class NetworkBlockOld(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.2, activate_before_residual=False):
        super(NetworkBlockOld, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate,
                                      activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate,
                                activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)