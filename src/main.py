# main.py
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import MolToSmiles
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import numpy as np
import pandas as pd

from layers import DrugADRDataset, collate_fn, ADR_STOI, atom_ftr
from models import DrugSEFusionTransformer


def split_dataset(df, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        random_state=seed,
        shuffle=True,
        stratify=df["LABEL"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (test_ratio + val_ratio),
        random_state=seed,
        shuffle=True,
        stratify=temp_df["LABEL"]
    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def make_dataloaders(train_df, val_df, test_df, batch_size=32):
    train_loader = DataLoader(
        DrugADRDataset(train_df),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        DrugADRDataset(val_df),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        DrugADRDataset(test_df),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for bg, adr_ids, adr_mask, labels in loader:
        bg = bg.to(device)
        adr_ids = adr_ids.to(device)
        adr_mask = adr_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(bg, adr_ids, adr_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, desc="Eval"):
    model.eval()
    all_logits = []
    all_labels = []

    for bg, adr_ids, adr_mask, labels in loader:
        bg = bg.to(device)
        adr_ids = adr_ids.to(device)
        adr_mask = adr_mask.to(device)
        labels = labels.to(device)

        logits = model(bg, adr_ids, adr_mask)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.float32)

    auc = roc_auc_score(labels, probs)
    aupr = average_precision_score(labels, probs)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds)

    print(f"{desc} - AUC: {auc:.4f}, AUPR: {aupr:.4f}, ACC: {acc:.4f}, F1: {f1:.4f}")
    return auc, aupr, acc, f1


def main():
    data_path = "D:\IT\OneDrive - Hanoi University of Science and Technology\Documents\HUST\Deep Learning\ADR_DL_Project\Dataset\ADR_with_SMILES_label.csv"  # chỉnh đường dẫn file CSV của bạn
    df = pd.read_csv(data_path)
    df = df[df.SMILES.apply(lambda x: Chem.MolFromSmiles(str(x)) is not None)]
    df = df.reset_index(drop=True)

    # đảm bảo LABEL là 0/1
    df["LABEL"] = df["LABEL"].astype(float)

    train_df, val_df, test_df = split_dataset(df)

    train_loader, val_loader, test_loader = make_dataloaders(train_df, val_df, test_df, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = DrugSEFusionTransformer(
        node_feat_dim=atom_ftr.feat_size(),
        adr_vocab_size=len(ADR_STOI) + 1,
        drug_max_nodes=64,
        adr_max_len=32,
        hidden_size=200
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    EPOCHS = 20
    best_val_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch:03d} - Train loss: {train_loss:.4f}")
        val_auc, val_aupr, _, _ = evaluate(model, val_loader, device, desc="Val")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "best_model.pt")
            print("  >> Saved best model.")

    print("Loading best model and evaluating on test set...")
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    evaluate(model, test_loader, device, desc="Test")


if __name__ == "__main__":
    main()
