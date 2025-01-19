import torch
import torch.nn as nn
from torchvision import models

from hourglass import Conv, Hourglass
from corner_pooling import TopLeftPool, BottomRightPool

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
    def __init__(self, num_stacks, in_channels, num_classes, backbone_name='resnet18'):
        """
        num_stacks: number of hourglass stacks
        in_channels: number of input channels
        num_classes: number of classes to predict
        """
        super(CornerNet, self).__init__()

        # pre-processing layer to extract features from the input image
        self.pre = nn.Sequential(
            Conv(in_channels, 128, kernel_size=7, stride=2, bn=True),
            nn.MaxPool2d(3, stride=2)
        )

        # hourglass network to extract features from the input image
        # cornetnet does not use the classic cnn backbone, but instead uses a hourglass network
        # the hourglass network is a stack of hourglass modules, each of which is a residual network
        self.hg = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, 256),
                Conv(256, 256, bn=True)
            ) for _ in range(num_stacks)
        ])

        # top left part of the model to retrieve the top left corner of the bounding box
        self.tl_modules = nn.ModuleList([
            nn.Sequential(
                TopLeftPool(),
                Conv(256, 256, bn=True)
            ) for _ in range(num_stacks)
        ])

        # bottom right part of the model to retrieve the bottom right corner of the bounding box
        self.br_modules = nn.ModuleList([
            nn.Sequential(
                BottomRightPool(),
                Conv(256, 256, bn=True)
            ) for _ in range(num_stacks)
        ])

        # prediction heads to predict the heatmap, the bounding box regression, and the corner regression
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

            # add the output of the hourglass network to the input image to get the next hourglass network input
            # this is used to create a residual connection between the input and the output of the hourglass network
            if i < len(self.hg) - 1:
                x = x + hg

        return outputs
