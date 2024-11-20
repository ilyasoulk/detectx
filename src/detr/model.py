import torch
import torchvision
import torch.nn as nn
import lightning as L


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
        # Self-Attention  
        x_qkv = self.w_qkv(x)
        Q, K, V = x_qkv.chunk(3, -1)
        Q, K, V = split_into_heads(Q, K, V, num_heads=self.num_heads)
        attn_out, _ = head_level_self_attention(Q, K, V)
        attn_out = concat_heads(attn_out)
        attn_out += x
        ln_1 = self.ln_1(attn_out)

        # FFN
        mlp_1 = self.mlp_1(ln_1)
        x = self.activation(mlp_1)
        x = self.mlp_2(x)
        x += attn_out
        x = self.ln_2(x)

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, fc_dim, num_heads, num_obj, activation="relu") -> None:
        super().__init__()
        self.object_emb = nn.Embedding(num_embeddings=num_obj, embedding_dim=hidden_dim)
        self.w_self_qkv = nn.Linear(hidden_dim, 3*hidden_dim)
        self.ln_1 = nn.LayerNorm(hidden_dim)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.w_kv = nn.Linear(hidden_dim, 2*hidden_dim)
        self.w_q = nn.Linear(hidden_dim, hidden_dim)

        self.ln_3 = nn.LayerNorm(hidden_dim)
        self.mlp_1 = nn.Linear(hidden_dim, fc_dim)
        self.mlp_2 = nn.Linear(fc_dim, hidden_dim)
        self.activation = getattr(nn.functional, activation)
        self.num_heads = num_heads
        self.num_obj = num_obj

    def forward(self, x):
        tgt = self.object_emb(torch.arange(self.num_obj, device=x.device))
        tgt = tgt.expand(x.size(0), -1, -1)

        # Self-Attention on objects
        x_qkv = self.w_self_qkv(tgt)
        Q, K, V = x_qkv.chunk(3, -1)
        # (B, 100, 256)
        Q, K, V = split_into_heads(Q, K, V, num_heads=self.num_heads)
        self_attn_out, _ = head_level_self_attention(Q, K, V)
        self_attn_out = concat_heads(self_attn_out)
        self_attn_out += tgt
        objects = self.ln_1(self_attn_out)


        # Cross-attention objects / encoder output
        x_kv = self.w_kv(x)
        K, V = x_kv.chunk(2, -1)
        Q = self.w_q(objects)
        Q, K, V = split_into_heads(Q, K, V, num_heads=self.num_heads)
        cross_attn_out, _ = head_level_self_attention(Q, K, V)
        cross_attn_out = concat_heads(cross_attn_out)
        cross_attn_out += tgt
        cross_attn_out = self.ln_2(cross_attn_out)

        # FFN
        mlp_1 = self.mlp_1(cross_attn_out)
        x = self.activation(mlp_1)
        x = self.mlp_2(x)
        x += cross_attn_out
        x = self.ln_3(x)

        return x



class PredictionHeads(nn.Module):
    def __init__(self, d, num_cls, input_dim) -> None:
        super().__init__()
        self.box_head = nn.Sequential(
            nn.Linear(input_dim, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        
        self.class_head = nn.Sequential(
            nn.Linear(input_dim, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        
        self.bbox_output = nn.Linear(d, 4)  # [xmin, ymin, xmax, ymax]
        self.class_output = nn.Linear(d, num_cls)

    def forward(self, x):
        box_head = self.box_head(x)
        class_head = self.class_head(x)

        bbox = self.bbox_output(box_head)
        classes = self.class_output(class_head)

        return classes, bbox



class DeTr(nn.Module):
    def __init__(self, backbone, hidden_dim, input_shape, fc_dim, num_heads, activ_fn, num_encoder, num_decoder, num_obj, d, num_cls) -> None:
        super().__init__()
        self.backbone = getattr(torchvision.models, backbone)(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove fully connected layers
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        # Outputs shape (B, hidden_dim, 16, 16) for input shape (3, 512, 512)
        self.output_shape = self._compute_output_shape(input_shape)
        _, _, H, W = self.output_shape
        self.seq_len = H*W
        self.pos_embedding = nn.Embedding(num_embeddings=(self.seq_len), embedding_dim=hidden_dim)

        self.encoder = nn.Sequential(
            *(TransformerEncoder(hidden_dim, fc_dim, num_heads, activ_fn) for _ in range(num_encoder))
        )
        self.decoder = nn.Sequential(
            *(TransformerDecoder(hidden_dim, fc_dim, num_heads, num_obj, activ_fn) for _ in range(num_decoder))
        )

        self.prediction_heads = PredictionHeads(d, num_cls, hidden_dim)


    def _compute_output_shape(self, input_shape):
        dummy_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            output = self.backbone(dummy_input)
        return output.shape

    def forward(self, x):
        # Backbone + Position Embedding + Transformer + Prediction Heads
        x = self.backbone(x)
        x = self.conv(x)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = x.transpose(-1, -2)
        pos_embedding = self.pos_embedding(torch.arange(self.seq_len, device=x.device))
        x += pos_embedding

        x = self.encoder(x)
        x = self.decoder(x)

        classes, bboxes = self.prediction_heads(x)

        return classes, bboxes





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
    d = 50
    num_cls = 20
    hidden_dim = 128
    model = DeTr(backbone, hidden_dim, input_shape, fc_dim, num_heads, activ_fn, num_encoder, num_decoder, num_obj, d, num_cls)
    x = torch.randn(1, *input_shape)
    x = x.to(device)
    model = model.to(device)
    classes, bboxes = model(x)
    print(classes.shape, bboxes.shape) # (1, num_obj, num_cls), (1, num_obj, 4)
