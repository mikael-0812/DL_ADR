import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoder import DrugEncoder, SEEncoder


# ============ BERT-Style Self-Attention ============

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key   = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        # x: (B, L, H) -> (B, heads, L, head_dim)
        B, L, H = x.size()
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        # Linear projections
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        # Split heads
        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask  # (B, 1, 1, L)

        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, V)  # (B, heads, L, head_dim)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), context.size(1), self.hidden_size)
        return context


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.act   = nn.GELU()

    def forward(self, x):
        return self.act(self.dense(x))


class BertOutput(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.dense   = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, x, residual):
        x = self.dense(x)
        x = self.dropout(x)
        return self.norm(x + residual)   # residual + LN


class BertLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn      = BertSelfAttention(hidden_size, num_heads, dropout)
        self.self_output    = BertOutput(hidden_size, hidden_size, dropout)
        self.intermediate   = BertIntermediate(hidden_size, intermediate_size)
        self.intermediate_o = BertOutput(intermediate_size, hidden_size, dropout)

    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        attn_out = self.self_attn(hidden_states, attention_mask)
        attn_out = self.self_output(attn_out, hidden_states)

        # FFN
        inter = self.intermediate(attn_out)
        out   = self.intermediate_o(inter, attn_out)
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


# ============ Bi-Directional Cross-Attention (Drug ↔ SE) ============

class BiDirectionalCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = hidden_size // num_heads

        # Drug → SE
        self.Q_drug = nn.Linear(hidden_size, hidden_size)
        self.K_se   = nn.Linear(hidden_size, hidden_size)
        self.V_se   = nn.Linear(hidden_size, hidden_size)

        # SE → Drug
        self.Q_se   = nn.Linear(hidden_size, hidden_size)
        self.K_drug = nn.Linear(hidden_size, hidden_size)
        self.V_drug = nn.Linear(hidden_size, hidden_size)

        self.dropout   = nn.Dropout(dropout)
        self.out_drug  = nn.Linear(hidden_size, hidden_size)
        self.out_se    = nn.Linear(hidden_size, hidden_size)

        self.norm_drug = nn.LayerNorm(hidden_size, eps=1e-12)
        self.norm_se   = nn.LayerNorm(hidden_size, eps=1e-12)

    def _split_heads(self, x):
        B, L, H = x.size()
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        B, heads, L, dim = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, L, heads * dim)

    def _attend(self, Q, K, V, mask):
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask   # (B,1,1,L_k)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, V)
        context = self._merge_heads(context)
        return context

    def forward(self, drug, se, drug_mask=None, se_mask=None):
        # Drug attends to SE
        Qd = self.Q_drug(drug)
        Ke = self.K_se(se)
        Ve = self.V_se(se)
        drug_ctx = self._attend(Qd, Ke, Ve, se_mask)
        drug_out = self.out_drug(drug_ctx)
        drug_out = self.norm_drug(drug_out + drug)  # residual

        # SE attends to Drug
        Qe = self.Q_se(se)
        Kd = self.K_drug(drug)
        Vd = self.V_drug(drug)
        se_ctx = self._attend(Qe, Kd, Vd, drug_mask)
        se_out = self.out_se(se_ctx)
        se_out = self.norm_se(se_out + se)

        return drug_out, se_out


class BiCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn_drug_to_se = BertSelfAttention(hidden_size, num_heads, dropout)
        self.attn_se_to_drug = BertSelfAttention(hidden_size, num_heads, dropout)

        self.output_d = BertOutput(hidden_size, hidden_size, dropout)
        self.output_s = BertOutput(hidden_size, hidden_size, dropout)

    def forward(self, drug_seq, se_seq, drug_mask, se_mask):
        # masks
        drug_attn_mask = (1 - drug_mask).unsqueeze(1).unsqueeze(2) * -10000.0
        se_attn_mask   = (1 - se_mask).unsqueeze(1).unsqueeze(2) * -10000.0

        # drug attends to SE
        drug2se = self.attn_drug_to_se(drug_seq, se_attn_mask)
        drug_out = self.output_d(drug2se, drug_seq)

        # SE attends to Drug
        se2drug = self.attn_se_to_drug(se_seq, drug_attn_mask)
        se_out = self.output_s(se2drug, se_seq)

        return drug_out, se_out

class DrugSEFusionTransformer(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        vocab_size,
        hidden_size=200
    ):
        super().__init__()

        self.drug_encoder = DrugEncoder(
            node_feat_dim=node_feat_dim,
            trans_hidden_dim=hidden_size
        )

        self.se_encoder = SEEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size
        )

        self.cross = BiCrossAttention(hidden_size, num_heads=4)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, bg, se_ids, se_mask):
        # Drug → GNN → Transformer
        drug_seq, drug_repr, drug_mask = self.drug_encoder(bg)

        # SE → Transformer
        se_seq, se_repr, se_mask = self.se_encoder(se_ids, se_mask)

        # Cross Attention Fusion
        drug_fused, se_fused = self.cross(drug_seq, se_seq, drug_mask, se_mask)

        # Final pooled vectors
        drug_final = drug_fused.mean(1)
        se_final = se_fused.mean(1)

        # Fusion
        fused = torch.cat([drug_final, se_final], dim=1)

        # Classification
        out = self.classifier(fused)
        return out