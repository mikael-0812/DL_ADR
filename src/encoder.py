import torch
from torch import nn

from layers import BertEncoder
from utils import SimpleAttentiveFPGNN


class GNN2TransformerEncoder(nn.Module):
    """
    Pipeline:
    DGLGraph (drug) --GNN(AttentiveFP style)--> node embeddings (N, gnn_hidden)
                    --pad per-graph--> (B, max_nodes, gnn_hidden)
                    --Linear--> (B, max_nodes, trans_hidden)
                    --+pos_emb--> TransformerEncoder
                    -> seq_out (B, max_nodes, trans_hidden)
                    -> graph_repr (B, trans_hidden) via masked mean pooling
    """

    def __init__(
        self,
        node_feat_dim,          # dim của node feature đầu vào (vd từ AttentiveFPAtomFeaturizer)
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

        self.max_nodes = max_nodes
        self.trans_hidden_dim = trans_hidden_dim

        # GNN kiểu AttentiveFP
        self.gnn = SimpleAttentiveFPGNN(
            in_dim=node_feat_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            num_timesteps=gnn_timesteps,
            num_heads=gnn_heads,
            dropout=dropout
        )

        # Chiếu node embedding GNN -> hidden size Transformer
        self.proj = nn.Linear(gnn_hidden_dim, trans_hidden_dim)

        # Positional embedding cho node thứ i trong graph
        self.pos_emb = nn.Embedding(max_nodes, trans_hidden_dim)

        # BERT-style Transformer Encoder
        self.encoder = BertEncoder(
            num_layers=trans_layers,
            hidden_size=trans_hidden_dim,
            intermediate_size=trans_intermediate,
            num_heads=trans_heads,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.pooler = nn.Linear(trans_hidden_dim, trans_hidden_dim)
        self.pool_act = nn.Tanh()

    def _build_attention_mask(self, mask):
        """
        mask: (B, max_nodes) 1 = valid, 0 = pad
        -> (B, 1, 1, max_nodes) thêm -10000 cho pad
        """
        if mask is None:
            return None
        extended = (1.0 - mask.float()).unsqueeze(1).unsqueeze(2)  # pad=1
        extended = extended * -10000.0
        return extended

    def _masked_mean_pooling(self, x, mask):
        """
        x: (B, L, H), mask: (B, L) với 1=valid, 0=pad
        """
        if mask is None:
            return x.mean(dim=1)
        mask = mask.float().unsqueeze(-1)  # (B,L,1)
        x = x * mask
        sum_x = x.sum(dim=1)    # (B,H)
        denom = mask.sum(dim=1) + 1e-8
        return sum_x / denom

    def forward(self, bg):
        """
        bg: batched DGLGraph
            - bg.ndata['h']: (N_total, node_feat_dim)
        """

        device = bg.device if hasattr(bg, "device") else bg.ndata["h"].device

        node_feats = bg.ndata['h']  # (N_total, node_feat_dim)

        # 1) GNN: node embeddings
        node_emb = self.gnn(bg, node_feats)  # (N_total, gnn_hidden_dim)

        # 2) Tách theo từng graph và pad/truncate tới max_nodes
        batch_num_nodes = bg.batch_num_nodes().tolist()  # list length B
        B = len(batch_num_nodes)

        max_nodes = self.max_nodes
        H_gnn = node_emb.size(1)

        seq_gnn = node_emb.new_zeros((B, max_nodes, H_gnn))
        mask = torch.zeros((B, max_nodes), dtype=torch.long, device=node_emb.device)

        start = 0
        for i, n_nodes in enumerate(batch_num_nodes):
            end = start + n_nodes
            h_i = node_emb[start:end]  # (n_nodes, H_gnn)

            if n_nodes >= max_nodes:
                seq_gnn[i, :, :] = h_i[:max_nodes]
                mask[i, :] = 1
            else:
                seq_gnn[i, :n_nodes, :] = h_i
                mask[i, :n_nodes] = 1

            start = end

        # 3) Project -> Transformer hidden + positional embedding
        x = self.proj(seq_gnn)  # (B, L, trans_hidden)

        L = x.size(1)
        positions = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)
        pos_embed = self.pos_emb(positions)  # (1,L,H)
        x = x + pos_embed

        x = self.dropout(x)

        # 4) Build attention mask & encode
        attn_mask = self._build_attention_mask(mask)  # (B,1,1,L)
        seq_out = self.encoder(x, attn_mask)          # (B,L,H)

        # 5) Graph-level pooling
        graph_repr = self._masked_mean_pooling(seq_out, mask)  # (B,H)
        graph_repr = self.pool_act(self.pooler(graph_repr))

        return seq_out, graph_repr, mask

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




