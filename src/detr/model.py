from typing import Tuple
import torch
import torchvision
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
    def __init__(self, backbone, input_shape) -> None:
        super().__init__()
        self.backbone = getattr(torchvision.models, backbone)(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove fully connected layers
        # Outputs shape (B, 2048, 16, 16) for input shape (3, 512, 512)
        self.output_shape = self._compute_output_shape(input_shape)
        _, C, H, W = self.output_shape
        self.seq_len = H*W
        self.pos_embedding = nn.Embedding(num_embeddings=(self.seq_len), embedding_dim=C)


    def _compute_output_shape(self, input_shape):
        dummy_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            output = self.backbone(dummy_input)
        return output.shape

    def forward(self, x):
        # Backbone + Embedding block
        x = self.backbone(x)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = x.transpose(-1, -2)
        pos_embedding = self.pos_embedding(torch.arange(self.seq_len, device=x.device))
        x += pos_embedding
        return x



if __name__ == "__main__":

    device = torch.device("mps")
    input_shape = (3, 512, 512)
    backbone = "resnet50"
    model = DeTr(backbone, input_shape)
    x = torch.randn(1, *input_shape)
    x = x.to(device)
    model = model.to(device)
    y = model(x)
    print(y.shape) # (1, 256, 2048)
