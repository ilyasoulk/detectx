import torch
import torchvision
import torch.nn as nn


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
    def __init__(self, hidden_dim, fc_dim, num_heads, activation="relu", dropout=0.1):
        # Input shape : (B, S, H)
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.w_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp_1 = nn.Linear(hidden_dim, fc_dim)
        self.mlp_2 = nn.Linear(fc_dim, hidden_dim)
        self.activation = getattr(nn.functional, activation)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        # Initialize weights with slightly larger variance
        nn.init.xavier_uniform_(self.w_qkv.weight, gain=1.414)
        nn.init.xavier_uniform_(self.mlp_1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.mlp_2.weight, gain=1.414)

    def forward(self, x):
        # Pre-norm architecture
        x_norm = self.ln_1(x)

        # Self-Attention
        x_qkv = self.w_qkv(x_norm)
        Q, K, V = x_qkv.chunk(3, -1)
        Q, K, V = split_into_heads(Q, K, V, num_heads=self.num_heads)
        attn_out, attn_weights = head_level_self_attention(Q, K, V)
        attn_out = concat_heads(attn_out)

        # Add dropout after attention
        attn_out = self.dropout(attn_out)
        x = x + attn_out

        # FFN with dropout
        x_norm = self.ln_2(x)
        mlp_out = self.mlp_1(x_norm)
        mlp_out = self.activation(mlp_out)
        mlp_out = self.dropout(mlp_out)  # Add dropout between FFN layers
        mlp_out = self.mlp_2(mlp_out)
        mlp_out = self.dropout(mlp_out)  # Add dropout after FFN

        x = x + mlp_out

        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        fc_dim,
        num_heads,
        num_obj,
        activation="relu",
        seq_len=256,
        dropout=0.1,
    ) -> None:
        super().__init__()
        self.object_emb = nn.Embedding(num_embeddings=num_obj, embedding_dim=hidden_dim)
        self.query_pos_emb = nn.Embedding(
            num_embeddings=num_obj, embedding_dim=hidden_dim
        )
        self.memory_pos_emb = nn.Embedding(
            num_embeddings=seq_len, embedding_dim=hidden_dim
        )
        self.w_self_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.ln_1 = nn.LayerNorm(hidden_dim)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.w_kv = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.w_q = nn.Linear(hidden_dim, hidden_dim)

        self.ln_3 = nn.LayerNorm(hidden_dim)
        self.mlp_1 = nn.Linear(hidden_dim, fc_dim)
        self.mlp_2 = nn.Linear(fc_dim, hidden_dim)
        self.activation = getattr(nn.functional, activation)
        self.num_heads = num_heads
        self.num_obj = num_obj
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Get object queries and their positions
        tgt = self.object_emb(torch.arange(self.num_obj, device=x.device))
        query_pos = self.query_pos_emb(torch.arange(self.num_obj, device=x.device))
        tgt = tgt + query_pos
        tgt = tgt.expand(x.size(0), -1, -1)
        memory_pos = self.memory_pos_emb(torch.arange(x.size(1), device=x.device))
        x = x + memory_pos.unsqueeze(0)

        # Self-Attention on object queries
        x_qkv = self.w_self_qkv(tgt)
        Q, K, V = x_qkv.chunk(3, -1)
        Q, K, V = split_into_heads(Q, K, V, num_heads=self.num_heads)
        self_attn_out, _ = head_level_self_attention(Q, K, V)
        self_attn_out = concat_heads(self_attn_out)
        self_attn_out = self.dropout(self_attn_out)  # Add dropout after self-attention
        self_attn_out += tgt
        objects = self.ln_1(self_attn_out)

        # Cross-attention between object queries and encoder memory
        x_kv = self.w_kv(x)
        K, V = x_kv.chunk(2, -1)
        Q = self.w_q(objects)

        # Add query positional embeddings to Q
        Q = Q + query_pos.unsqueeze(0)

        Q, K, V = split_into_heads(Q, K, V, num_heads=self.num_heads)
        cross_attn_out, _ = head_level_self_attention(Q, K, V)
        cross_attn_out = concat_heads(cross_attn_out)
        cross_attn_out = self.dropout(
            cross_attn_out
        )  # Add dropout after cross-attention
        cross_attn_out += objects
        cross_attn_out = self.ln_2(cross_attn_out)

        # FFN
        mlp_1 = self.mlp_1(cross_attn_out)
        x = self.activation(mlp_1)
        x = self.dropout(x)  # Add dropout between FFN layers
        x = self.mlp_2(x)
        x = self.dropout(x)  # Add dropout after FFN
        x += cross_attn_out
        x = self.ln_3(x)

        return x


class PredictionHeads(nn.Module):
    def __init__(self, num_cls, hidden_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # Add dropout between layers
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # Add dropout between layers
        )

        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # Add dropout between layers
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # Add dropout between layers
        )

        self.bbox_output = nn.Linear(hidden_dim, 4)  # [xmin, ymin, xmax, ymax]
        self.class_output = nn.Linear(hidden_dim, num_cls)

    def forward(self, x):
        box_head = self.box_head(x)
        class_head = self.class_head(x)

        bbox = self.bbox_output(box_head)
        classes = self.class_output(class_head)

        return classes, bbox


class DeTr(nn.Module):
    def __init__(
        self,
        backbone,
        hidden_dim,
        input_shape,
        fc_dim,
        num_heads,
        activ_fn,
        num_encoder,
        num_decoder,
        num_obj,
        num_cls,
        dropout=0.1,
    ) -> None:
        super().__init__()
        self.backbone = getattr(torchvision.models, backbone)(pretrained=True)
        self.backbone = nn.Sequential(
            *list(self.backbone.children())[:-2]
        )  # Remove fully connected layers
        print(sum([param.numel() for param in self.backbone.parameters()]))
        self.output_shape = self._compute_output_shape(input_shape)
        _, C, H, W = self.output_shape
        self.conv = nn.Conv2d(C, hidden_dim, 1)
        # Outputs shape (B, hidden_dim, 16, 16) for input shape (3, 512, 512)
        self.seq_len = H * W
        self.pos_embedding = nn.Embedding(
            num_embeddings=(self.seq_len), embedding_dim=hidden_dim
        )

        self.encoder = nn.Sequential(
            *(
                TransformerEncoder(hidden_dim, fc_dim, num_heads, activ_fn, dropout)
                for _ in range(num_encoder)
            )
        )
        self.decoder = nn.Sequential(
            *(
                TransformerDecoder(
                    hidden_dim,
                    fc_dim,
                    num_heads,
                    num_obj,
                    activ_fn,
                    self.seq_len,
                    dropout,
                )
                for _ in range(num_decoder)
            )
        )

        self.prediction_heads = PredictionHeads(num_cls, hidden_dim, dropout)

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
