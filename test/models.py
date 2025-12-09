import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

from chem.loader import mol_to_graph_data_obj_simple


def smiles2graph(smiles: str):
    """Convert SMILES → PyG graph for SimSGT"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    g = mol_to_graph_data_obj_simple(mol)
    g.smiles = smiles
    return g

class MILAttention(nn.Module):
    """
    MIL Attention pooling cho tập token (subgraph embeddings) của 1 phân tử.
    Input:  token_emb: (N_i, d_token)
    Output: drug_emb: (d_token,)
            att_weights: (N_i,)
    """
    def __init__(self, d_token, d_att=128):
        super().__init__()
        self.fc_a = nn.Linear(d_token, d_att)
        self.fc_b = nn.Linear(d_att, 1)

    def forward(self, token_emb):
        # token_emb: (N, d)
        # score_i = w^T tanh(W h_i)
        h = torch.tanh(self.fc_a(token_emb))             # (N, d_att)
        scores = self.fc_b(h).squeeze(-1)               # (N,)
        att_weights = torch.softmax(scores, dim=0)      # (N,)
        drug_emb = torch.sum(att_weights.unsqueeze(-1) * token_emb, dim=0)  # (d,)
        return drug_emb, att_weights

from transformers import AutoTokenizer, AutoModel

class ADRTextEncoder(nn.Module):
    """
    Encode ADR_TERM (chuỗi text) thành vector embedding.
    Dùng BERT-based model. Mặc định: Bio_ClinicalBERT.
    """
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

    def forward(self, adr_text_list):
        """
        adr_text_list: list[str] length = B
        return: (B, hidden_size)
        """
        enc = self.tokenizer(
            adr_text_list,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(self.bert.device)
        attention_mask = enc["attention_mask"].to(self.bert.device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Lấy CLS embedding
        cls_emb = outputs.last_hidden_state[:, 0, :]   # (B, hidden_size)
        return cls_emb

class DrugEncoderSimSGT(nn.Module):
    """
    Dùng SimSGT (TokenMAE) để trích xuất token embeddings (subgraph-level)
    + MIL Attention → drug-level embedding.
    """
    def __init__(self, simsgt_model, d_token=300, d_mil_att=128, device="cuda"):
        super().__init__()
        self.simsgt = simsgt_model.to(device)
        self.simsgt.eval()
        self.mil_pool = MILAttention(d_token=d_token, d_att=d_mil_att)
        self.device = device

    @torch.no_grad()
    def extract_tokens(self, smiles: str):
        data = smiles2graph(smiles)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        data.mask_tokens = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.x_masked = data.x.clone()

        data = data.to(self.device)

        token_emb = self.simsgt.tokenizer(
            data.x_masked,
            data.edge_index,
            data.edge_attr
        )  # (N, 300)
        return token_emb

    @torch.no_grad()
    def encode_batch(self, smiles_list):
        """
        smiles_list: list[str], length = B
        return:
          drug_embs: (B, d_token)
          att_weights_list: list[Tensor(N_i,)]
        """
        drug_embs = []
        att_ws = []

        for smi in smiles_list:
            token_emb = self.extract_tokens(smi)   # (N_i, d_token)
            drug_emb_i, att_i = self.mil_pool(token_emb)
            drug_embs.append(drug_emb_i)
            att_ws.append(att_i)

        drug_embs = torch.stack(drug_embs, dim=0)  # (B, d_token)
        return drug_embs, att_ws

class ADRSeverityModel(nn.Module):
    """
    Model cuối:
      - Drug encoder: SimSGT + MIL
      - ADR encoder: BERT
      - Head: MLP → 5 class (severity 1–5)
    """
    def __init__(
        self,
        drug_encoder: DrugEncoderSimSGT,
        adr_encoder: ADRTextEncoder,
        d_hidden=256,
        n_classes=5,
        device="cuda"
    ):
        super().__init__()
        self.drug_encoder = drug_encoder
        self.adr_encoder = adr_encoder
        self.device = device

        d_drug = drug_encoder.mil_pool.fc_a.in_features   # 300
        d_adr = adr_encoder.hidden_size                   # ~768

        # Project về cùng dimension
        self.proj_drug = nn.Linear(d_drug, d_hidden)
        self.proj_adr  = nn.Linear(d_adr,  d_hidden)

        # Combine [z_d, z_a, z_d * z_a]
        self.classifier = nn.Sequential(
            nn.Linear(3 * d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, n_classes)
        )

    def forward(self, smiles_list, adr_text_list):
        """
        smiles_list: list[str] length = B
        adr_text_list: list[str] length = B
        Output: logits (B, n_classes)
        """
        # 1) Drug embedding từ SimSGT + MIL
        drug_embs, _ = self.drug_encoder.encode_batch(smiles_list)   # (B, d_drug)

        # 2) ADR embedding từ BERT
        adr_embs = self.adr_encoder(adr_text_list)                   # (B, d_adr)

        # 3) Project về space chung
        drug_z = self.proj_drug(drug_embs)   # (B, d_hidden)
        adr_z  = self.proj_adr(adr_embs)     # (B, d_hidden)

        # 4) Kết hợp
        interaction = drug_z * adr_z         # (B, d_hidden)
        h = torch.cat([drug_z, adr_z, interaction], dim=-1)  # (B, 3*d_hidden)

        logits = self.classifier(h)          # (B, n_classes)
        return logits
