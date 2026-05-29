
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader


#  CHEMIN

base_dir = r"C:\Users\CE PC1\Downloads\Doc de materiaux\output\output_corriges"
output_dir = os.path.dirname(base_dir)


#  LECTURE ET NETTOYAGE

def read_csv_robust(path):
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass

    raise ValueError(f"Impossible de lire correctement le fichier : {path}")


def clean_columns(df):
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def normalize_column_names_nodes(df):
    rename_map = {}
    for col in df.columns:
        low = col.lower().strip()
        if low in ["grain_id", "grainid", "id", "node_id", "nodeid"]:
            rename_map[col] = "grain_id"
        elif low == "x":
            rename_map[col] = "x"
        elif low == "y":
            rename_map[col] = "y"
        elif low in ["r", "radius"]:
            rename_map[col] = "R"
    return df.rename(columns=rename_map)


def normalize_column_names_forces(df):
    rename_map = {}
    for col in df.columns:
        low = col.lower().strip()
        if low in ["contact", "contact_id"]:
            rename_map[col] = "contact"
        elif low in ["grain_i", "graini", "i"]:
            rename_map[col] = "grain_i"
        elif low in ["grain_j", "grainj", "j"]:
            rename_map[col] = "grain_j"
        elif low in ["wall_pos", "wall", "wall_position"]:
            rename_map[col] = "wall_pos"
        elif low in ["fij", "force", "forceij"]:
            rename_map[col] = "fij"
    return df.rename(columns=rename_map)


def check_required_columns(df, required_cols, file_label):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {file_label} : {missing}\n"
            f"Colonnes trouvées : {df.columns.tolist()}"
        )


# 3) CONSTRUCTION DE GRAPHE

def build_graph(nodes_path, forces_path):
    nodes = read_csv_robust(nodes_path)
    forces = read_csv_robust(forces_path)

    nodes = clean_columns(nodes)
    forces = clean_columns(forces)

    nodes = normalize_column_names_nodes(nodes)
    forces = normalize_column_names_forces(forces)

    check_required_columns(nodes, ["grain_id", "x", "y", "R"], "nodes")
    check_required_columns(forces, ["grain_i", "grain_j", "wall_pos", "fij"], "forces")

    nodes["grain_id"] = pd.to_numeric(nodes["grain_id"], errors="coerce")
    nodes["x"] = pd.to_numeric(nodes["x"], errors="coerce")
    nodes["y"] = pd.to_numeric(nodes["y"], errors="coerce")
    nodes["R"] = pd.to_numeric(nodes["R"], errors="coerce")

    nodes = nodes.dropna(subset=["grain_id", "x", "y", "R"]).copy()
    nodes["grain_id"] = nodes["grain_id"].astype(int)

    forces["grain_i"] = pd.to_numeric(forces["grain_i"], errors="coerce")
    forces["fij"] = pd.to_numeric(forces["fij"], errors="coerce")
    forces["grain_j"] = forces["grain_j"].astype(str).str.strip()
    forces["wall_pos"] = forces["wall_pos"].astype(str).str.strip().str.lower()

    forces = forces.dropna(subset=["grain_i", "fij"]).copy()
    forces["grain_i"] = forces["grain_i"].astype(int)

    grain_ids = nodes["grain_id"].tolist()
    id_to_idx = {gid: i for i, gid in enumerate(grain_ids)}

    # Features noeuds : [x, y, R, is_grain, is_wall, wall_code]
    node_features = []
    for _, row in nodes.iterrows():
        node_features.append([
            float(row["x"]),
            float(row["y"]),
            float(row["R"]),
            1.0,
            0.0,
            0.0
        ])

    wall_code_map = {"left": -1.0, "right": 1.0, "top": 2.0}
    present_walls = []

    for w in forces["wall_pos"].dropna().astype(str).str.lower():
        if w in wall_code_map and w not in present_walls:
            present_walls.append(w)

    wall_to_idx = {}
    for wall in present_walls:
        wall_to_idx[wall] = len(node_features)
        node_features.append([
            0.0, 0.0, 0.0,
            0.0, 1.0, wall_code_map[wall]
        ])

    x = torch.tensor(node_features, dtype=torch.float)

    src, dst, edge_y = [], [], []

    for _, row in forces.iterrows():
        grain_i = int(row["grain_i"])
        fij = float(row["fij"])

        if grain_i not in id_to_idx:
            continue

        i_idx = id_to_idx[grain_i]
        grain_j_raw = str(row["grain_j"]).strip()
        grain_j_num = pd.to_numeric(grain_j_raw, errors="coerce")

        if pd.notna(grain_j_num):
            grain_j = int(grain_j_num)
            if grain_j not in id_to_idx:
                continue
            j_idx = id_to_idx[grain_j]
        else:
            wall = str(row["wall_pos"]).strip().lower()
            if wall not in wall_to_idx:
                continue
            j_idx = wall_to_idx[wall]

        src.append(i_idx)
        dst.append(j_idx)
        edge_y.append([fij])

        src.append(j_idx)
        dst.append(i_idx)
        edge_y.append([fij])

    if len(src) == 0:
        raise ValueError("Aucune arête valide trouvée dans ce graphe.")

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_y = torch.tensor(edge_y, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    data.edge_y = edge_y
    return data


# 4) CHARGEMENT DE PLUSIEURS GRAPHES

def load_graphs(base_dir, start_idx, end_idx):
    graphs = []
    valid_ids = []

    for i in range(start_idx, end_idx + 1):
        nodes_path = os.path.join(base_dir, f"nodes_{i}.csv")
        forces_path = os.path.join(base_dir, f"forces_{i}.csv")

        if not os.path.exists(nodes_path):
            print(f"[IGNORÉ] nodes manquant : {nodes_path}")
            continue

        if not os.path.exists(forces_path):
            print(f"[IGNORÉ] forces manquant : {forces_path}")
            continue

        try:
            data = build_graph(nodes_path, forces_path)
            graphs.append(data)
            valid_ids.append(i)
            print(f"[OK] fichier {i} chargé")
        except Exception as e:
            print(f"[ERREUR] fichier {i} : {e}")

    return graphs, valid_ids


# 5) NORMALISATION

def compute_normalization_stats(graphs):
    all_x = torch.cat([g.x for g in graphs], dim=0)
    all_y = torch.cat([g.edge_y for g in graphs], dim=0)

    x_mean = all_x.mean(dim=0, keepdim=True)
    x_std = all_x.std(dim=0, keepdim=True)
    x_std[x_std == 0] = 1.0

    y_mean = all_y.mean(dim=0, keepdim=True)
    y_std = all_y.std(dim=0, keepdim=True)
    y_std[y_std == 0] = 1.0

    return x_mean, x_std, y_mean, y_std


def apply_normalization(graphs, x_mean, x_std, y_mean, y_std):
    norm_graphs = []

    for g in graphs:
        g2 = copy.deepcopy(g)
        g2.x = (g2.x - x_mean) / x_std
        g2.edge_y = (g2.edge_y - y_mean) / y_std
        norm_graphs.append(g2)

    return norm_graphs


#  MODELE (reseau neuronal de graphe)

class EdgeForceGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))

        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]

        edge_feat = torch.cat([h_src, h_dst], dim=1)
        pred = self.edge_mlp(edge_feat)
        return pred


#  ENTRAINEMENT

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch.x, batch.edge_index)
        loss = criterion(pred, batch.edge_y)
        mae = F.l1_loss(pred, batch.edge_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mae += mae.item()
        total_batches += 1

    return total_loss / total_batches, total_mae / total_batches


#  TRACER DES CHAINES DE FORCES

def plot_force_chains_comparison(
    raw_graph,
    pred_real,
    true_real,
    title_pred="Chaînes de force prédites (GNN)",
    title_true="Chaînes de force vraies (DEM)",
    force_threshold_ratio=0.7,
    save_path=None
):
    x = raw_graph.x.cpu().numpy()
    edge_index = raw_graph.edge_index.cpu().numpy()

    node_x = x[:, 0]
    node_y = x[:, 1]
    node_r = x[:, 2]
    is_grain = x[:, 3]
    is_wall = x[:, 4]
    wall_code = x[:, 5]

    pred_real = pred_real.detach().cpu().numpy().flatten()
    true_real = true_real.detach().cpu().numpy().flatten()

    # On garde une seule arête par contact
    unique_edges = []
    seen = set()

    for k in range(edge_index.shape[1]):
        i = int(edge_index[0, k])
        j = int(edge_index[1, k])

        key = tuple(sorted((i, j)))
        if key in seen:
            continue

        seen.add(key)
        unique_edges.append((k, i, j))

    pred_vals = np.array([pred_real[k] for k, _, _ in unique_edges])
    true_vals = np.array([true_real[k] for k, _, _ in unique_edges])

    pred_max = pred_vals.max() if len(pred_vals) > 0 else 0.0
    true_max = true_vals.max() if len(true_vals) > 0 else 0.0

    pred_thr = force_threshold_ratio * pred_max
    true_thr = force_threshold_ratio * true_max

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, values, threshold, vmax, title in zip(
        axes,
        [true_vals, pred_vals],
        [true_thr, pred_thr],
        [true_max, pred_max],
        [title_true, title_pred]
    ):
        # Importance des grains = force max des arêtes connectées
        node_importance = np.zeros(len(node_x))

        for idx_u, (k, i, j) in enumerate(unique_edges):
            fval = values[idx_u]
            node_importance[i] = max(node_importance[i], fval)
            node_importance[j] = max(node_importance[j], fval)

        # Normalisation entre 0 et 1
        if node_importance.max() > 0:
            node_importance = node_importance / node_importance.max()

        # Grains colorés selon l'importance dans les chaînes de force
        for i in range(len(node_x)):
            if is_grain[i] > 0.5:
                importance = node_importance[i]

                if importance > 0.6:
                    color = "black"
                elif importance > 0.3:
                    color = "0.7"
                else:
                    color = "white"

                circ = Circle(
                    (node_x[i], node_y[i]),
                    node_r[i],
                    facecolor=color,
                    edgecolor="black",
                    linewidth=2.0
                )
                ax.add_patch(circ)

        # Chaînes de force
        for idx_u, (k, i, j) in enumerate(unique_edges):
            fval = values[idx_u]

            if fval < threshold:
                continue

            xi, yi = node_x[i], node_y[i]
            xj, yj = node_x[j], node_y[j]

            # Si contact avec une paroi
            if is_wall[j] > 0.5:
                code = wall_code[j]
                if code == -1.0:      # left
                    xj, yj = 0.0, yi
                elif code == 1.0:     # right
                    xj, yj = 100.0, yi
                elif code == 2.0:     # top
                    xj, yj = xi, 100.0

            elif is_wall[i] > 0.5:
                code = wall_code[i]
                if code == -1.0:
                    xi, yi = 0.0, yj
                elif code == 1.0:
                    xi, yi = 100.0, yj
                elif code == 2.0:
                    xi, yi = xj, 100.0

            lw = 3 + 4.0 * (fval / vmax) if vmax > 0 else 3

            ax.plot([xi, xj], [yi, yj], color="red", linewidth=lw, alpha=0.95)

        ax.set_title(title, fontsize=14)
        ax.set_aspect("equal")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# PROGRAMME PRINCIPAL : ENTRAINEMENT SEULEMENT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

raw_train_graphs, train_ids = load_graphs(base_dir, 0, 499)

print("\nRÉSUMÉ CHARGEMENT TRAIN")
print("Nombre de graphes train :", len(raw_train_graphs))
print("IDs train :", train_ids)

if len(raw_train_graphs) == 0:
    raise ValueError("Aucun graphe d'entraînement chargé.")

train_graphs = copy.deepcopy(raw_train_graphs)

x_mean, x_std, y_mean, y_std = compute_normalization_stats(train_graphs)
train_graphs = apply_normalization(train_graphs, x_mean, x_std, y_mean, y_std)

train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)

model = EdgeForceGNN(
    in_channels=train_graphs[0].x.shape[1],
    hidden_channels=64
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("\nDÉBUT DE L'ENTRAÎNEMENT")

for epoch in range(1, 101):
    train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, criterion, device)

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss = {train_loss:.6f} | Train MAE = {train_mae:.6f}"
        )

torch.save(model.state_dict(), "edge_force_gnn_train_only.pth")
print("\nModèle sauvegardé : edge_force_gnn_train_only.pth")


#  TEST VISUEL SUR LES GRAPHES D'ENTRAINEMENT

model.eval()

sample_idx = 10  # tu peux changer ici pour les fichiers

sample_norm = train_graphs[sample_idx].to(device)
sample_raw = raw_train_graphs[sample_idx]

with torch.no_grad():
    pred = model(sample_norm.x, sample_norm.edge_index)

    mse = F.mse_loss(pred, sample_norm.edge_y).item()
    mae = F.l1_loss(pred, sample_norm.edge_y).item()

    print("\nRESULTATS SUR UN GRAPHE D'ENTRAINEMENT")
    print(f"MSE normalisé : {mse:.6f}")
    print(f"MAE normalisé : {mae:.6f}")

    pred_real = pred * y_std.to(device) + y_mean.to(device)
    true_real = sample_norm.edge_y * y_std.to(device) + y_mean.to(device)

    print("\nCOMPARAISON SUR 100 ARETES")
    for i in range(min(100, pred_real.shape[0])):
        vrai = true_real[i].item()
        p = pred_real[i].item()
        print(f"Arête {i:03d} | vrai = {vrai:.6f} | prédit = {p:.6f}")

plot_force_chains_comparison(
    raw_graph=sample_raw,
    pred_real=pred_real,
    true_real=true_real,
    title_true="Chaînes de force vraies (DEM)",
    title_pred="Chaînes de force prédites (GNN) - Train",
    force_threshold_ratio=1e-12,
    save_path=os.path.join(output_dir, "comparaison_dem_gnn_entrainement.png")
)

print("\nImage sauvegardée : comparaison_dem_gnn_train.png")
