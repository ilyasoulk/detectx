import torch
import torch.nn as nn


"""
Tuple structure: (kernel_size, filters, stride, padding)"
"M" -> Maxpooling with stride 2x2 and kernel 2x2
List structure: [tuple, tuple, repeat_nb]
"""

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) # not in the original paper
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.bn(self.conv(x)))

class YOLOv1(nn.Module):
    def __init__(self, input_shape=(3, 448, 448), **kwargs):
        super().__init__()
        self.architecture = architecture_config
        self.input_shape = input_shape
        
        self.darknet = self._make_conv_layers(self.architecture)
        self.fc_layers = self._make_fc_layers(**kwargs)
        
    def forward(self, x):
        assert x.shape[1:] == self.input_shape # (C, H, W)
        
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x
        
    def _make_conv_layers(self, architecture):
        layers = []
        in_channels = self.input_shape[0]

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels=in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                repeat_nb = x[2]

                for _ in range(repeat_nb):
                    layers += [
                        CNNBlock(
                            in_channels=in_channels,
                            out_channels=conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            in_channels=conv1[1],
                            out_channels=conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)
        
    def _make_fc_layers(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Linear(1024 * S * S, 496), # 496 for testing, original is 4096
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (B * 5 + C)), # 496 for testing, original is 4096
        )
