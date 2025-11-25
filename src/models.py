
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


# ======= BERT-style building blocks =======

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        B, L, H = x.size()
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (B, heads, L, head_dim)

    def forward(self, hidden_states, attention_mask=None):
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask  # (B,1,1,L)

        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, V)  # (B,heads,L,head_dim)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), context.size(1), self.hidden_size)
        return context


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.dense(x))


class BertOutput(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, x, residual):
        x = self.dense(x)
        x = self.dropout(x)
        return self.norm(x + residual)


class BertLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = BertSelfAttention(hidden_size, num_heads, dropout)
        self.self_output = BertOutput(hidden_size, hidden_size, dropout)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.intermediate_o = BertOutput(intermediate_size, hidden_size, dropout)

    def forward(self, hidden_states, attention_mask=None):
        attn_out = self.self_attn(hidden_states, attention_mask)
        attn_out = self.self_output(attn_out, hidden_states)

        inter = self.intermediate(attn_out)
        out = self.intermediate_o(inter, attn_out)
        return out


class BertEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, intermediate_size, num_heads, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            BertLayer(hidden_size, intermediate_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


# ======= Simple AttentiveFP-like GNN =======

from dgl.nn.pytorch import GATConv


class SimpleAttentiveFPGNN(nn.Module):
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
        assert hidden_dim % num_heads == 0

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.num_heads = num_heads

        self.node_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList([
            GATConv(
                in_feats=hidden_dim,
                out_feats=hidden_dim // num_heads,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                residual=True,
                activation=F.elu,
                allow_zero_in_degree = True
            )
            for _ in range(num_layers)
        ])

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, g, node_feats):
        h = self.node_proj(node_feats)  # (N, hidden_dim)

        for layer in range(self.num_layers):
            h_in = h
            h = self.convs[layer](g, h_in)  # (N, heads, dim_per_head)
            h = h.view(h.shape[0], -1)  # (N, hidden_dim)

        # Timesteps GRU (approx AttentiveFP)
        h_seq = h.unsqueeze(1)  # (N,1,H)
        for _ in range(self.num_timesteps):
            h_seq, _ = self.gru(h_seq)
        h = h_seq.squeeze(1)  # (N,H)

        return h


# ======= Drug Encoder: GNN → sequence → Transformer =======

class GNN2TransformerDrugEncoder(nn.Module):
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
        self.max_nodes = max_nodes
        self.trans_hidden_dim = trans_hidden_dim

        self.gnn = SimpleAttentiveFPGNN(
            in_dim=node_feat_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            num_timesteps=gnn_timesteps,
            num_heads=gnn_heads,
            dropout=dropout
        )

        self.proj = nn.Linear(gnn_hidden_dim, trans_hidden_dim)
        self.pos_emb = nn.Embedding(max_nodes, trans_hidden_dim)

        self.encoder = BertEncoder(
            num_layers=trans_layers,
            hidden_size=trans_hidden_dim,
            intermediate_size=trans_intermediate,
            num_heads=trans_heads,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

    def _build_attention_mask(self, mask):
        if mask is None:
            return None
        # mask: (B,L) 1=valid,0=pad → bias: (B,1,1,L)
        return (1.0 - mask.float()).unsqueeze(1).unsqueeze(2) * -10000.0

    def forward(self, bg: dgl.DGLGraph):
        node_feats = bg.ndata["h"]  # (N_total, node_feat_dim)
        node_emb = self.gnn(bg, node_feats)  # (N_total, gnn_hidden_dim)

        batch_num_nodes = bg.batch_num_nodes().tolist()
        B = len(batch_num_nodes)
        L = self.max_nodes
        H_gnn = node_emb.size(1)

        seq = node_emb.new_zeros((B, L, H_gnn))
        mask = torch.zeros((B, L), dtype=torch.long, device=node_emb.device)

        start = 0
        for i, n in enumerate(batch_num_nodes):
            end = start + n
            h_i = node_emb[start:end]  # (n,H)
            if n >= L:
                seq[i] = h_i[:L]
                mask[i] = 1
            else:
                seq[i, :n] = h_i
                mask[i, :n] = 1
            start = end

        # proj + pos
        x = self.proj(seq)  # (B,L,Ht)
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos)
        x = self.dropout(x)

        attn_bias = self._build_attention_mask(mask)
        seq_out = self.encoder(x, attn_bias)  # (B,L,Ht)

        return seq_out, mask  # không pool ở đây, để cross-attention dùng


# ======= ADR / SE Encoder: text → Transformer =======

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
        self.hidden_size = hidden_size
        self.max_len = max_len

        self.emb = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_size)

        self.encoder = BertEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def _build_attention_mask(self, mask):
        if mask is None:
            return None
        return (1.0 - mask.float()).unsqueeze(1).unsqueeze(2) * -10000.0

    def forward(self, ids, mask):
        """
        ids: (B,L)
        mask: (B,L) 1=valid,0=pad
        """
        B, L = ids.size()
        pos = torch.arange(L, device=ids.device).unsqueeze(0)

        x = self.emb(ids) + self.pos_emb(pos)
        x = self.dropout(x)

        attn_bias = self._build_attention_mask(mask)
        seq_out = self.encoder(x, attn_bias)  # (B,L,H)

        return seq_out, mask


# ======= Bi-directional Cross-Attention Drug ↔ SE =======

class BiDirectionalCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Drug → SE
        self.Q_drug = nn.Linear(hidden_size, hidden_size)
        self.K_se = nn.Linear(hidden_size, hidden_size)
        self.V_se = nn.Linear(hidden_size, hidden_size)

        # SE → Drug
        self.Q_se = nn.Linear(hidden_size, hidden_size)
        self.K_drug = nn.Linear(hidden_size, hidden_size)
        self.V_drug = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.out_drug = nn.Linear(hidden_size, hidden_size)
        self.out_se = nn.Linear(hidden_size, hidden_size)

        self.norm_drug = nn.LayerNorm(hidden_size, eps=1e-12)
        self.norm_se = nn.LayerNorm(hidden_size, eps=1e-12)

    def _split_heads(self, x):
        B, L, H = x.size()
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (B,h,L,d)

    def _merge_heads(self, x):
        B, h, L, d = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, L, h * d)

    def _build_bias(self, mask, Lk):
        if mask is None:
            return None
        if mask.dim() == 4:
            return mask
        assert mask.dim() == 2
        assert mask.size(1) == Lk
        return (1.0 - mask.float()).unsqueeze(1).unsqueeze(2) * -10000.0

    def _attend(self, Q, K, V, mask_bias):
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask_bias is not None:
            scores = scores + mask_bias  # (B,1,1,L_k) broadcast

        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, V)  # (B,h,L_q,d)
        context = self._merge_heads(context)
        return context

    def forward(self, drug_seq, se_seq, drug_mask=None, se_mask=None):
        B, Ld, H = drug_seq.size()
        B2, Ls, H2 = se_seq.size()
        assert B == B2 and H == H2

        se_bias = self._build_bias(se_mask, Ls)
        drug_bias = self._build_bias(drug_mask, Ld)

        # Drug attends to SE
        Qd = self.Q_drug(drug_seq)
        Ke = self.K_se(se_seq)
        Ve = self.V_se(se_seq)
        drug_ctx = self._attend(Qd, Ke, Ve, se_bias)
        drug_out = self.out_drug(drug_ctx)
        drug_out = self.norm_drug(drug_out + drug_seq)

        # SE attends to Drug
        Qe = self.Q_se(se_seq)
        Kd = self.K_drug(drug_seq)
        Vd = self.V_drug(drug_seq)
        se_ctx = self._attend(Qe, Kd, Vd, drug_bias)
        se_out = self.out_se(se_ctx)
        se_out = self.norm_se(se_out + se_seq)

        return drug_out, se_out


# ======= Full Fusion Model =======

class DrugSEFusionTransformer(nn.Module):
    def __init__(
            self,
            node_feat_dim,
            adr_vocab_size,
            drug_max_nodes=64,
            adr_max_len=32,
            hidden_size=200
    ):
        super().__init__()

        self.drug_encoder = GNN2TransformerDrugEncoder(
            node_feat_dim=node_feat_dim,
            max_nodes=drug_max_nodes,
            trans_hidden_dim=hidden_size
        )

        self.se_encoder = SEEncoder(
            vocab_size=adr_vocab_size,
            hidden_size=hidden_size,
            max_len=adr_max_len
        )

        self.cross = BiDirectionalCrossAttention(hidden_size, num_heads=4)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)  # logits
        )

    def _masked_mean_pool(self, x, mask):
        # x: (B,L,H), mask: (B,L) with 1/0
        mask = mask.float().unsqueeze(-1)  # (B,L,1)
        x = x * mask
        summed = x.sum(1)  # (B,H)
        denom = mask.sum(1).clamp(min=1e-8)
        return summed / denom

    def forward(self, bg, adr_ids, adr_mask):
        # Drug side
        drug_seq, drug_mask = self.drug_encoder(bg)  # (B,Ld,H), (B,Ld)

        # ADR side
        se_seq, se_mask = self.se_encoder(adr_ids, adr_mask)  # (B,Ls,H),(B,Ls)

        # Cross-attention
        drug_fused, se_fused = self.cross(drug_seq, se_seq, drug_mask, se_mask)

        # Pool
        drug_repr = self._masked_mean_pool(drug_fused, drug_mask)
        se_repr = self._masked_mean_pool(se_fused, se_mask)

        fused = torch.cat([drug_repr, se_repr], dim=-1)
        logits = self.classifier(fused)
        return logits.squeeze(-1)  # (B,)
