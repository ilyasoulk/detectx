import torch
import torchvision
import torch.nn as nn

class YOLOv1(nn.Module):
    def __init__(self, num_classes, input_shape=(3, 448, 448), cell_dim=7, num_anchors=2):
        super().__init__()
        self.num_classes = num_classes
        self.cell_dim = cell_dim
        self.num_anchors = num_anchors
        self.num_output = num_anchors * 5 + num_classes
        self.input_shape = input_shape
        
        self.conv_layers = self.make_conv_layers()
        self.fc_layers = self.make_fc_layers()
        
    def make_conv_layers(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(192, 128, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, 2, 1),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.LeakyReLU(0.1),
            
        )
        
    def make_fc_layers(self):
        return nn.Sequential(
            nn.Linear(1024 * self.cell_dim * self.cell_dim, 4096),
            nn.Linear(4096, self.cell_dim * self.cell_dim * self.num_output),
        )
        
    def forward(self, x):
        print(x.shape[1:])
        print(self.input_shape)
        assert x.shape[1:] == self.input_shape # (C, H, W)
        
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = x.view(-1, self.cell_dim, self.cell_dim, self.num_output)
        return x
