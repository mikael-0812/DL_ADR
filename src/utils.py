import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GATConv
import math


class SimpleAttentiveFPGNN(nn.Module):
    """
    GNN kiểu AttentiveFP đơn giản:
    - node init MLP
    - nhiều lớp attention (GATConv)
    - GRU để cập nhật node features
    - trả về node embeddings (cho tất cả node trong batch graph)
    """

    def __init__(
        self,
        in_dim,
        hidden_dim=128,
        num_layers=3,
        num_timesteps=2,
        num_heads=4,
        dropout=0.1
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.num_heads = num_heads

        # Init node embedding
        self.node_proj = nn.Linear(in_dim, hidden_dim)

        # GAT-like attentive propagation
        self.convs = nn.ModuleList([
            GATConv(
                in_feats=hidden_dim,
                out_feats=hidden_dim // num_heads,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                residual=True,
                activation=F.elu
            )
            for _ in range(num_layers)
        ])

        # GRU update on node states
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, g, node_feats):
        """
        g: batched DGLGraph
        node_feats: (num_nodes_total, in_dim) = g.ndata['h']
        """
        h = self.node_proj(node_feats)  # (N, hidden_dim)

        # DGL expects (N, hidden_dim)
        for layer in range(self.num_layers):
            h_in = h
            # GATConv output: (N, num_heads, hidden_dim_per_head)
            h = self.convs[layer](g, h_in)
            # Merge heads
            h = h.view(h.shape[0], -1)  # (N, hidden_dim)

        # AttentiveFP có "timesteps" với GRU; ta approximate trên toàn batch:
        # Để dùng GRU, ta coi mỗi node như một "sequence length = 1"
        # nên ta reshape (N, hidden_dim) -> (N, 1, hidden_dim)
        h_seq = h.unsqueeze(1)
        for _ in range(self.num_timesteps):
            # GRU expects (batch, seq_len, hidden)
            h_seq, _ = self.gru(h_seq)
        h = h_seq.squeeze(1)  # (N, hidden_dim)

        return h  # node embeddings
