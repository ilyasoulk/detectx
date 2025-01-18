import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=not bn)
        self.bn = nn.BatchNorm2d(out_dim) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Hourglass(nn.Module):
    def __init__(self, n, f):
        super(Hourglass, self).__init__()
        for i in range(n):
            self.add_module('up{}'.format(i), Conv(f, f, bn=True))
            self.add_module('low{}'.format(i), Conv(f, f, bn=True))
        if n > 1:
            self.add_module('low1', Hourglass(n - 1, f))
        else:
            self.add_module('low1', Conv(f, f, bn=True))
        self.low2 = Conv(f, f, bn=True)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up = x
        for i in range(len(self._modules) // 2):
            up = self._modules['up{}'.format(i)](up)

        low = nn.MaxPool2d(2, 2)(x)
        for i in range(len(self._modules) // 2):
            low = self._modules['low{}'.format(i)](low)
        if 'low1' in self._modules:
            low = self._modules['low1'](low)
        low = self.low2(low)
        up = self.up(low)

        return up + x
