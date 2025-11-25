import dgl
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import string
from dgllife.utils import (
    smiles_to_bigraph,
    AttentiveFPAtomFeaturizer,
    AttentiveFPBondFeaturizer
)
from sklearn.model_selection import train_test_split

from src.models import DrugSEFusionTransformer


def split_dataset(df, seed=42,
                  train_ratio=0.7,
                  val_ratio=0.15,
                  test_ratio=0.15):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Train + temp
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        random_state=seed,
        shuffle=True,
        stratify=df["LABEL"],   # giữ phân bố label đều nhau
    )

    # temp -> val + test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_ratio / (test_ratio + val_ratio)),
        random_state=seed,
        shuffle=True,
        stratify=temp_df["LABEL"]
    )

    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))

    return train_df, val_df, test_df

ADR_CHARS = list(string.ascii_letters + string.digits + "-_'()[]+/ ")
ADR_STOI = {c: i+1 for i, c in enumerate(ADR_CHARS)}  # 0 = PAD
ADR_ITOS = {i+1: c for i, c in enumerate(ADR_CHARS)}

MAX_ADR_LEN = 32

def tokenize_adr(text):
    text = text[:MAX_ADR_LEN].lower()
    ids = []
    for c in text:
        ids.append(ADR_STOI.get(c, 0))  # unknown → 0
    if len(ids) < MAX_ADR_LEN:
        ids += [0] * (MAX_ADR_LEN - len(ids))
    mask = [1 if x!=0 else 0 for x in ids]
    return ids, mask

node_ftr = AttentiveFPAtomFeaturizer()
bond_ftr = AttentiveFPBondFeaturizer(self_loop=True)

def smiles_to_graph(smiles):
    g = smiles_to_bigraph(
        smiles,
        node_featurizer=node_ftr,
        edge_featurizer=bond_ftr,
        add_self_loop=True
    )
    return g

def collate_fn(samples):
    """
    samples: list of (g, adr_ids, adr_mask, label)
    """
    graphs, adr_ids, adr_mask, labels = map(list, zip(*samples))

    bg = dgl.batch(graphs)
    adr_ids = torch.stack(adr_ids, dim=0)
    adr_mask = torch.stack(adr_mask, dim=0)
    labels = torch.stack(labels, dim=0)

    return bg, adr_ids, adr_mask, labels

class DrugADRDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Graph from SMILES
        g = smiles_to_graph(row["SMILES"])
        g.ndata['h'] = g.ndata['h'].float()

        # ADR
        ids, mask = tokenize_adr(row["ADR_TERM"])

        label = torch.tensor([float(row["LABEL"])], dtype=torch.float32)

        return g, torch.tensor(ids), torch.tensor(mask), label

data = pd.read_csv("Dataset/ADR_with_SMILES_label.csv")
train_df, val_df, test_df = split_dataset(data)

train_loader = DataLoader(
    DrugADRDataset(train_df),
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)
test_loader = DataLoader(
    DrugADRDataset(test_df),
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    criterion = nn.BCELoss()

    for bg, adr_ids, adr_mask, labels in loader:
        bg = bg.to(device)
        adr_ids = adr_ids.to(device)
        adr_mask = adr_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(bg, adr_ids, adr_mask)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for bg, adr_ids, adr_mask, labels in loader:
            bg = bg.to(device)
            adr_ids = adr_ids.to(device)
            adr_mask = adr_mask.to(device)

            preds = model(bg, adr_ids, adr_mask)

            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.numpy())

    preds_arr = np.array(preds_list).flatten()
    labels_arr = np.array(labels_list).flatten()

    auc = roc_auc_score(labels_arr, preds_arr)
    aupr = average_precision_score(labels_arr, preds_arr)

    return auc, aupr

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DrugSEFusionTransformer(
    node_feat_dim=node_ftr.feat_size(),
    vocab_size=len(ADR_STOI)+1,
    hidden_size=200
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 20

for epoch in range(1, EPOCHS+1):
    loss = train_one_epoch(model, train_loader, optimizer, device)
    auc, aupr = evaluate(model, test_loader, device)

    print(f"Epoch {epoch} | Loss={loss:.4f} | AUC={auc:.4f} | AUPR={aupr:.4f}")
