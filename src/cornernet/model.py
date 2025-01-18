import torch
import torch.nn as nn

from .hourglass import Conv, Hourglass
from .corner_pooling import TopLeftPool, BottomRightPool

class PredictionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PredictionHead, self).__init__()
        self.conv1 = Conv(in_dim, 256, bn=True)
        self.conv2 = nn.Conv2d(256, out_dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class CornerNet(nn.Module):
    def __init__(self, num_stacks, in_channels, num_classes):
        super(CornerNet, self).__init__()
        self.pre = nn.Sequential(
            Conv(in_channels, 128, kernel_size=7, stride=2, bn=True),
            nn.MaxPool2d(3, stride=2)
        )
        self.hg = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, 256),
                Conv(256, 256, bn=True)
            ) for _ in range(num_stacks)
        ])
        self.tl_modules = nn.ModuleList([
            nn.Sequential(
                TopLeftPool(),
                Conv(256, 256, bn=True)
            ) for _ in range(num_stacks)
        ])
        self.br_modules = nn.ModuleList([
            nn.Sequential(
                BottomRightPool(),
                Conv(256, 256, bn=True)
            ) for _ in range(num_stacks)
        ])
        self.tl_heats = nn.ModuleList([PredictionHead(256, num_classes) for _ in range(num_stacks)])
        self.br_heats = nn.ModuleList([PredictionHead(256, num_classes) for _ in range(num_stacks)])
        self.tl_regrs = nn.ModuleList([PredictionHead(256, 2) for _ in range(num_stacks)])
        self.br_regrs = nn.ModuleList([PredictionHead(256, 2) for _ in range(num_stacks)])

    def forward(self, x):
        x = self.pre(x)
        outputs = []
        for i in range(len(self.hg)):
            hg = self.hg[i](x)
            tl = self.tl_modules[i](hg)
            br = self.br_modules[i](hg)

            tl_heat = self.tl_heats[i](tl)
            br_heat = self.br_heats[i](br)
            tl_regr = self.tl_regrs[i](tl)
            br_regr = self.br_regrs[i](br)
            outputs.append([tl_heat, br_heat, tl_regr, br_regr])
            if i < len(self.hg) - 1:
                x = x + hg
        return outputs
