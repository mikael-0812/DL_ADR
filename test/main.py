import torch
import torch.nn.functional as F
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from chem.model import TokenMAE
from models import DrugEncoderSimSGT, ADRTextEncoder, ADRSeverityModel


# ============================================================
# 1. Dummy args để mô hình hoạt động (vì checkpoint không chứa args)
# ============================================================
class DummyArgs:
    def __init__(self):
        # GNN parameters
        self.gnn_emb_dim = 300
        self.gnn_dropout = 0.0
        self.gnn_JK = "last"
        self.gnn_type = "gin"
        self.gnn_activation = "relu"
        self.decoder_jk = "last"

        # Transformer parameters
        self.d_model = 128
        self.dim_feedforward = 512
        self.nhead = 4
        self.transformer_dropout = 0.0
        self.transformer_activation = "relu"
        self.transformer_norm_input = True
        self.custom_trans = True
        self.drop_mask_tokens = True
        self.use_trans_decoder = False   # IMPORTANT for checkpoint

        # Encoder layers
        self.gnn_token_layer = 1
        self.gnn_encoder_layer = 5
        self.trans_encoder_layer = 4

        # Decoder layers
        self.gnn_decoder_layer = 3
        self.decoder_input_norm = False
        self.trans_decoder_layer = 0

        # General configs
        self.nonpara_tokenizer = False
        self.moving_average_decay = 0.99
        self.loss = "mse"
        self.loss_all_nodes = False
        self.subgraph_mask = False
        self.zero_mask = False
        self.eps = 0.5

        # Position Encoding
        self.pe_type = "none"
        self.laplacian_norm = "none"
        self.max_freqs = 20
        self.eigvec_norm = "L2"
        self.raw_norm_type = "none"
        self.kernel_times = []
        self.kernel_times_func = "none"
        self.layers = 3
        self.post_layers = 2
        self.dim_pe = 28
        self.phi_hidden_dim = 32
        self.phi_out_dim = 32

def make_dummy_args():
    return DummyArgs()



# ============================================================
# 2. Hàm load pretrained SimSGT (m35_ckt.pt)
# ============================================================
def load_simsgt_model(checkpoint_path="/content/SimSGT/chem/checkpoints/m35_ckt.pt"):

    args = make_dummy_args()

    model = TokenMAE(
        gnn_encoder_layer=args.gnn_encoder_layer,
        gnn_token_layer=args.gnn_token_layer,
        gnn_decoder_layer=args.gnn_decoder_layer,
        gnn_emb_dim=args.gnn_emb_dim,
        nonpara_tokenizer=args.nonpara_tokenizer,
        gnn_JK=args.gnn_JK,
        gnn_dropout=args.gnn_dropout,
        gnn_type=args.gnn_type,

        d_model=args.d_model,
        trans_encoder_layer=args.trans_encoder_layer,
        trans_decoder_layer=args.trans_decoder_layer,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        transformer_dropout=args.transformer_dropout,
        transformer_activation=F.relu,
        transformer_norm_input=args.transformer_norm_input,
        custom_trans=args.custom_trans,
        drop_mask_tokens=args.drop_mask_tokens,
        use_trans_decoder=args.use_trans_decoder,

        pe_type=args.pe_type,
        args=args,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt  # checkpoint chỉ là state_dict

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Loaded pretrained m35_ckt.pt")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    model.eval()
    return model

class SmilesADRFreqDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.smiles = df["SMILES"].astype(str).tolist()
        self.adr = df["ADR_TERM"].astype(str).tolist()
        # LABEL ∈ {1,...,5} → chuyển về {0,...,4} cho CrossEntropy
        self.labels = (df["LABEL"].astype(int) - 1).tolist()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.adr[idx], self.labels[idx]

def collate_fn(batch):
    smiles_list = [b[0] for b in batch]
    adr_list = [b[1] for b in batch]
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return smiles_list, adr_list, labels

import torch
import torch.nn.functional as F
from torch.optim import Adam

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load SimSGT pretrained
    simsgt_model = load_simsgt_model()  # dùng hàm bạn đã có
    drug_encoder = DrugEncoderSimSGT(simsgt_model, d_token=300, d_mil_att=128, device=device)

    # 2) ADR encoder (BERT)
    adr_encoder = ADRTextEncoder("emilyalsentzer/Bio_ClinicalBERT").to(device)

    # 3) Model cuối
    model = ADRSeverityModel(
        drug_encoder=drug_encoder,
        adr_encoder=adr_encoder,
        d_hidden=256,
        n_classes=5,
        device=device
    ).to(device)

    # 4) Dataset + Dataloader
    train_dataset = SmilesADRFreqDataset("data.csv")
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2):  # demo 2 epoch
        total_loss = 0.0
        for smiles_batch, adr_batch, labels in train_loader:
            labels = labels.to(device)

            # forward
            logits = model(smiles_batch, adr_batch)  # (B, 5)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}")

    # 5) Test nhanh 1 sample
    model.eval()
    test_smi = ["CCN(CC)CC"]
    test_adr = ["rash"]
    with torch.no_grad():
        logits = model(test_smi, test_adr)
        probs = F.softmax(logits, dim=-1)
        pred_cls = probs.argmax(dim=-1).item()  # 0..4
    print("Pred class (0-based):", pred_cls, "→ severity =", pred_cls + 1)


if __name__ == "__main__":
    main()
