from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Shape of Pascal VOC [3, 442, 500]

def split_into_heads(Q, K, V, num_heads):
    Q = Q.reshape(Q.shape[0], Q.shape[1], num_heads, -1)
    K = K.reshape(K.shape[0], K.shape[1], num_heads, -1)
    V = V.reshape(V.shape[0], V.shape[1], num_heads, -1)
    return Q, K, V


def head_level_self_attention(Q, K, V):
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)
    d = Q.shape[-1]


    A = (Q @ K.transpose(-1, -2) / d**0.5).softmax(-1)
    attn_out = A @ V
    return attn_out.transpose(1, 2), A


def concat_heads(input_tensor):
  return input_tensor.flatten(-2, -1)

class TransformerEncoder(nn.Module):
    pass


class TransformerDecoder(nn.Module):
    pass



class DeTr(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove fully connected layers
        # Outputs shape (B, 2048, 14, 16)

    def forward(self, x):
        x = self.backbone(x)
        print(x.shape)


if __name__ == "__main__":

# Define basic transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

# Download PASCAL VOC 2012
# Set download=True to download if not already present
# By default downloads to ./data/VOCdevkit/VOC2012/
    dataset = datasets.VOCDetection(
        root='../../data',  # where to save the dataset
        year='2012',    # can use '2007' or '2012'
        image_set='train',  # can be 'train', 'val', or 'trainval'
        download=True,
        transform=transform
    )

    x, y= dataset[0]
    device = torch.device("mps")
    model = DeTr()
    model = model.to(device)
    x = x.to(device)
    x = x.unsqueeze(dim=0)
    print(x.shape)
    y = model(x)
    



