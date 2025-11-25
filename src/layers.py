
import string
import torch
from torch.utils.data import Dataset, DataLoader
import dgl
from dgllife.utils import (
    smiles_to_bigraph,
    AttentiveFPAtomFeaturizer,
    AttentiveFPBondFeaturizer,
)

import pandas as pd

# ===== ADR tokenizer (char-level) =====
ADR_CHARS = list(string.ascii_letters + string.digits + "-_'()[]+/ " )
ADR_STOI = {c: i+1 for i, c in enumerate(ADR_CHARS)}  # 0 = PAD
ADR_ITOS = {i+1: c for i, c in enumerate(ADR_CHARS)}
ADR_MAX_LEN = 32

def tokenize_adr(text: str):
    text = (text or "").lower().strip()
    text = text[:ADR_MAX_LEN]
    ids = [ADR_STOI.get(c, 0) for c in text]
    if len(ids) < ADR_MAX_LEN:
        ids += [0] * (ADR_MAX_LEN - len(ids))
    mask = [1 if t != 0 else 0 for t in ids]
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


# ===== SMILES → DGL graph (AttentiveFP) =====
atom_ftr = AttentiveFPAtomFeaturizer()
bond_ftr = AttentiveFPBondFeaturizer(self_loop=False)

def smiles_to_graph(smiles: str):
    g = smiles_to_bigraph(
        smiles,
        node_featurizer=atom_ftr,
        edge_featurizer=bond_ftr,
        add_self_loop=False
    )
    g.ndata["h"] = g.ndata["h"].float()
    return g


# ===== Dataset =====

class DrugADRDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smi = row["SMILES"]
        adr = row["ADR_TERM"]
        label = float(row["LABEL"])

        try:
            g = smiles_to_graph(smi)
            if g is None:
                # skip invalid sample or assign placeholder graph
                return self.__getitem__((idx + 1) % len(self.df))

        except Exception as e:
            print("Bad SMILES:", smi)
            raise e

        adr_ids, adr_mask = tokenize_adr(adr)
        y = torch.tensor(label, dtype=torch.float32)

        return g, adr_ids, adr_mask, y


def collate_fn(samples):
    graphs, adr_ids, adr_masks, labels = map(list, zip(*samples))
    bg = dgl.batch(graphs)
    adr_ids  = torch.stack(adr_ids, dim=0)
    adr_masks = torch.stack(adr_masks, dim=0)
    labels   = torch.stack(labels, dim=0)
    return bg, adr_ids, adr_masks, labels
