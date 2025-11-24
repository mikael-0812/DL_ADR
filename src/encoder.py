import torch
from torch import nn

from src.layers import BertEncoder
from src.models import GNN2TransformerEncoder


class DrugEncoder(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        gnn_hidden_dim=128,
        gnn_layers=3,
        gnn_timesteps=2,
        gnn_heads=4,
        trans_hidden_dim=200,
        trans_layers=4,
        trans_intermediate=512,
        trans_heads=8,
        max_nodes=64,
        dropout=0.1
    ):
        super().__init__()
        self.encoder = GNN2TransformerEncoder(
            node_feat_dim=node_feat_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_layers=gnn_layers,
            gnn_timesteps=gnn_timesteps,
            gnn_heads=gnn_heads,
            trans_hidden_dim=trans_hidden_dim,
            trans_layers=trans_layers,
            trans_intermediate=trans_intermediate,
            trans_heads=trans_heads,
            max_nodes=max_nodes,
            dropout=dropout
        )

    def forward(self, bg):
        seq_out, graph_repr, mask = self.encoder(bg)
        return seq_out, graph_repr, mask

class SEEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size=200,
        max_len=32,
        num_layers=4,
        num_heads=8,
        intermediate_size=512,
        dropout=0.1
    ):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_len, hidden_size)

        self.encoder = BertEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            dropout=dropout
        )

        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.act = nn.Tanh()

    def forward(self, ids, mask):
        B, L = ids.size()
        pos = torch.arange(0, L, device=ids.device).unsqueeze(0)

        x = self.emb(ids) + self.pos_emb(pos)
        attn_mask = (1 - mask).unsqueeze(1).unsqueeze(2) * -10000.0

        seq_out = self.encoder(x, attn_mask)

        pooled = (seq_out * mask.unsqueeze(-1)).sum(1) / (mask.sum(1).unsqueeze(-1) + 1e-8)

        pooled = self.act(self.pooler(pooled))

        return seq_out, pooled, mask




