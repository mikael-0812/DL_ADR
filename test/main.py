import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import Dataset, DataLoader
from chem.model import TokenMAE
from models import DrugEncoderSimSGT, ADRTextEncoder, ADRSeverityModel, smiles2graph

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
def load_simsgt_model(checkpoint_path="chem/checkpoints/m35_ckt.pt"):

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

@torch.no_grad()
def extract_simsgt_tokens(smiles, model):
    """
    Trích token sau Transformer Encoder
    """
    data = smiles2graph(smiles)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data.mask_tokens = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.x_masked = data.x.clone()

    # 1) Tokenizer → GNN
    h = model.tokenizer(data.x_masked, data.edge_index, data.edge_attr)

    # 2) Position encoding
    pe = model.pos_encoder(data)

    # 3) Transformer encoder (contextualized token)
    h = model.encoder(
        F.relu(h),
        data.edge_index,
        data.edge_attr,
        data.batch,
        data.mask_tokens,
        pe
    )

    # h shape = (num_nodes, 300)
    return h

def collect_all_embeddings(smiles_list, model):
    embeds = []
    node_map = []   # giữ track node → (smiles, node_index)

    for smi in smiles_list:
        h = extract_simsgt_tokens(smi, model)
        embeds.append(h)
        for i in range(h.shape[0]):
            node_map.append((smi, i))

    embeds = torch.cat(embeds, dim=0)   # [N_total_nodes, 300]
    return embeds, node_map

def build_codebook(all_embeds, n_clusters=128):
    km = KMeans(
        n_clusters=n_clusters,
        random_state=0,
        n_init="auto"
    )
    km.fit(all_embeds.numpy())

    centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    return km, centers

def molecule_to_vector(smiles, model, km):
    tokens = extract_simsgt_tokens(smiles, model)   # [n_nodes, 300]

    labels = km.predict(tokens.numpy())
    n_clusters = km.n_clusters

    vec = np.bincount(labels, minlength=n_clusters)
    vec = vec / (vec.sum() + 1e-6)   # chuẩn hóa thành phân phối

    return vec


from rdkit import Chem
from rdkit.Chem import Draw


def visualize_cluster_center(cluster_id, centers, embeds, node_map, top_k=1):
    center = centers[cluster_id]  # 300-d

    dists = torch.norm(embeds - center, dim=1)  # N distances
    top_idx = torch.topk(dists, k=top_k, largest=False).indices

    figs = []
    for idx in top_idx:
        smi, node_id = node_map[idx]
        mol = Chem.MolFromSmiles(smi)
        atom_list = [node_id]
        img = Draw.MolToImage(mol, highlightAtoms=atom_list)
        figs.append(img)

    return figs

if __name__ == "__main__":
    model = load_simsgt_model()

    smiles_list = [
        "CCO",
        "CCN(CC)CC",
        "c1ccccc1"
    ]

    print("Collecting contextualized tokens…")
    all_tokens, node_map = collect_all_embeddings(smiles_list, model)

    print("Building KMeans codebook…")
    km, centers = build_codebook(all_tokens, n_clusters=32)

    print("Embedding molecules…")
    for smi in smiles_list:
        vec = molecule_to_vector(smi, model, km)
        print(smi, vec[:10], "...")

    print("Visualizing cluster 0:")
    imgs = visualize_cluster_center(0, centers, all_tokens, node_map)
    imgs[0].show()
