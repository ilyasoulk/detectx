from typing import Tuple
import torch
from torch.nn.modules import activation
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
    def __init__(self, hidden_dim, fc_dim, num_heads, activation="relu"):
        # Input shape : (B, S, H)
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.w_qkv = nn.Linear(hidden_dim, 3*hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp_1 = nn.Linear(hidden_dim, fc_dim)
        self.mlp_2 = nn.Linear(fc_dim, hidden_dim)
        self.activation = getattr(nn.functional, activation)
        self.num_heads = num_heads

    def forward(self, x):
        ln_1 = self.ln_1(x)
        x_qkv = self.w_qkv(ln_1)
        Q, K, V = x_qkv.chunk(3, -1)
        Q, K, V = split_into_heads(Q, K, V, num_heads=self.num_heads)
        attn_out, _ = head_level_self_attention(Q, K, V)
        attn_out = concat_heads(attn_out)
        attn_out += x

        ln_2 = self.ln_2(attn_out)
        mlp_1 = self.mlp_1(ln_2)
        x = self.activation(mlp_1)
        x = self.mlp_2(x)
        x += attn_out

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, fc_dim, num_heads, num_obj, activation="relu") -> None:
        super().__init__()
        self.object_emb = nn.Embedding(num_embeddings=num_obj, embedding_dim=hidden_dim)
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.w_kv = nn.Linear(hidden_dim, 2*hidden_dim)
        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp_1 = nn.Linear(hidden_dim, fc_dim)
        self.mlp_2 = nn.Linear(fc_dim, hidden_dim)
        self.activation = getattr(nn.functional, activation)
        self.num_heads = num_heads
        self.num_obj = num_obj

    def forward(self, x):
        tgt = self.object_emb(torch.arange(self.num_obj, device=x.device))
        tgt = self.ln_1(tgt)
        x_kv = self.w_kv(x)
        K, V = x_kv.chunk(2, -1)
        Q = self.w_q(tgt).unsqueeze(0)
        print(f"Q shape = {Q.shape}, V shape = {V.shape}, K shape = {K.shape}")
        Q, K, V = split_into_heads(Q, K, V, num_heads=self.num_heads)
        attn_out, _ = head_level_self_attention(Q, K, V)
        attn_out = concat_heads(attn_out)
        print(f"Attention shape = {attn_out.shape}")
        attn_out += tgt

        ln_2 = self.ln_2(attn_out)
        mlp_1 = self.mlp_1(ln_2)
        x = self.activation(mlp_1)
        x = self.mlp_2(x)
        x += attn_out

        return x



class PredictionHeads(nn.Module):
    def __init__(self, num_cls, bbox_features) -> None:
        super().__init__()


class DeTr(nn.Module):
    def __init__(self, backbone, input_shape, fc_dim, num_heads, activ_fn, num_encoder, num_decoder, num_obj) -> None:
        super().__init__()
        self.backbone = getattr(torchvision.models, backbone)(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove fully connected layers
        # Outputs shape (B, 2048, 16, 16) for input shape (3, 512, 512)
        self.output_shape = self._compute_output_shape(input_shape)
        _, C, H, W = self.output_shape
        self.seq_len = H*W
        self.pos_embedding = nn.Embedding(num_embeddings=(self.seq_len), embedding_dim=C)

        self.encoder = nn.Sequential(
            *(TransformerEncoder(C, fc_dim, num_heads, activ_fn) for _ in range(num_encoder))
        )
        self.decoder = nn.Sequential(
            *(TransformerDecoder(C, fc_dim, num_heads, num_obj, activ_fn) for _ in range(num_decoder))
        )


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
        x = self.encoder(x)
        x = self.decoder(x)


        return x



if __name__ == "__main__":

    device = torch.device("mps")
    input_shape = (3, 512, 512)
    backbone = "resnet50"
    num_encoder = 2
    activ_fn = "relu"
    num_heads = 8
    fc_dim = 256
    num_decoder = 2
    num_obj = 128
    model = DeTr(backbone, input_shape, fc_dim, num_heads, activ_fn, num_encoder, num_decoder, num_obj)
    x = torch.randn(1, *input_shape)
    x = x.to(device)
    model = model.to(device)
    y = model(x)
    print(y.shape) # (1, 256, 2048)
