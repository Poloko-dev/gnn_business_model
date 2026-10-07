"""Business location intelligence for Lesotho with the dual-mode GraphSAGE model of gnn_location_model.ipynb.

Run it with no arguments for an interactive session: the model is loaded once and you choose what to do.

    python predict_poi.py                                   interactive menu
    python predict_poi.py site --lat -29.315 --lon 27.487   what this location suits (no other input needed)
    python predict_poi.py suggest --category Retail --lat -29.315 --lon 27.487 --radius-km 2
                                                            best spots for a category around a location
    python predict_poi.py gaps --area Mafeteng              market gaps of the sub-districts matching a name or code
    python predict_poi.py new --lat -29.315 --lon 27.487 --name --hours
                                                            a POI described as it would be mapped in OSM
    python predict_poi.py existing --query "node/4340647091"   (node_id, osm_id or part of a name)
    python predict_poi.py batch my_sites.csv [--known] --out my_predictions.csv
    python predict_poi.py --check                           verify features and predictions against the notebook

One model, two modes (gnn_location_model.ipynb):
  site mode   the model sees only the location and its neighbourhood (densities, distances, towns, the shape and
              mapping profile of the surroundings, the categories of the nearest POIs). This is the mode for a
              candidate site where no business stands yet. Test macro-F1 0.245 on areas the model never saw
              (2x chance); the true category is among its 3 most likely in 72% of cases.
  known mode  the model also sees how the POI itself is mapped (name, address, opening hours, building outline).
              Test macro-F1 0.356; true category in the top 3 in 83% of cases.

How a location is scored. The model is a graph model, so a location is added to the graph of the 6,083 existing
POIs as one more node and its features are computed the way poi_eda.ipynb and poi_feature_engineering.ipynb compute
them. A candidate site (site mode) is a one-way node: it receives messages from its k = 8 nearest POIs but changes
nothing for the existing ones. A new POI in known mode is fully inserted, as a newly mapped place would be. All 25
trained models (5 CV splits x 5 seeds) score it, each with the training labels of its split visible as neighbour
categories, and their temperature-scaled probabilities are averaged. Label reuse, which the notebook selected on
validation, is not applied here: it changed test macro-F1 by less than 0.004.
"""
import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="IProgress not found")
warnings.filterwarnings("ignore", message=".*torch.jit.script.*")
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from pyproj import Transformer
from torch_geometric.nn import GATv2Conv, GCNConv, MessagePassing
from torch_geometric.utils import k_hop_subgraph, softmax, to_undirected

HERE = Path(__file__).resolve().parent
PROCESSED_CSV = HERE / "processed_poi.csv"
ENGINEERED_CSV = HERE / "poi_engineered.csv"
RAW_CSV = HERE / "points_of_interest.csv"
OUT = HERE / "model_outputs_v2"
MODEL_PT = OUT / "gnn_dual_models.pt"
METRICS_JSON = OUT / "metrics.json"
OOF_CSV = OUT / "oof_predictions.csv"
UNL_CSV = OUT / "unlabelled_predictions.csv"
GAPS_CSV = OUT / "market_gaps.csv"                 # written by gnn_location_model.ipynb, Section 9

K, KMAX = 8, 32                              # k-NN graph of the model; neighbourhood size of the ctx_ features
TOWN_MIN_POIS, TOWN_RADIUS_M = 50, 2000      # EDA town rule
N_FOLDS = 5
MODES = ("site", "known")
BBOX = (27.0115, -30.6756, 29.4557, -28.5706)   # Lesotho (lon_min, lat_min, lon_max, lat_max), from README.txt
TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32735", always_xy=True)   # UTM 35S, as in poi_eda.ipynb
TO_LL = Transformer.from_crs("EPSG:32735", "EPSG:4326", always_xy=True)

# Raw EDA column -> engineered column; True where the feature notebook applies log1p
RAW_NAMES = {
    "x_km": ("loc_x_km", False), "y_km": ("loc_y_km", False), "dist_town_km": ("loc_log_dist_town_km", True),
    "dist_maseru_km": ("loc_dist_maseru_km", False), "is_urban": ("loc_is_urban", False),
    "density_500m": ("loc_log_density_500m", True), "density_2km": ("loc_log_density_2km", True),
    "density_10km": ("loc_log_density_10km", True), "nn1_dist_m": ("loc_log_nn1_dist_m", True),
    "knn8_mean_dist_m": ("loc_log_knn8_mean_dist_m", True), "latrines_1km": ("loc_log_latrines_1km", True),
    "has_name": ("attr_has_name", False), "has_address": ("attr_has_address", False),
    "has_hours": ("attr_has_hours", False), "has_footprint": ("attr_has_footprint", False),
    "log_footprint_m2": ("attr_log_footprint_m2", False), "is_point": ("attr_is_point", False),
    "quality_score": ("attr_quality_score", False), "osm_id_rank": ("attr_osm_id_rank", False),
}
ATTRS = ["has_name", "has_address", "has_hours", "has_footprint", "log_footprint_m2", "is_point", "quality_score",
         "osm_id_rank"]


# --------------------------------------------------------------------------------------------------------------------
# Model (identical to gnn_location_model.ipynb, Section 2; dropout is inactive at prediction time)
# --------------------------------------------------------------------------------------------------------------------
class WSAGEConv(MessagePassing):
    """GraphSAGE layer with a weighted neighbour mean: lin_nbr(sum_j w_ji x_j) + lin_root(x_i)."""

    def __init__(self, d_in, d_out):
        super().__init__(aggr="add")
        self.lin_nbr = nn.Linear(d_in, d_out)
        self.lin_root = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x, edge_index, edge_weight):
        return self.lin_nbr(self.propagate(edge_index, x=x, w=edge_weight)) + self.lin_root(x)

    def message(self, x_j, w):
        return w.unsqueeze(1) * x_j


class GNN(nn.Module):
    def __init__(self, in_dim, n_classes, conv="sage", hidden=64, layers=2, heads=4, attn_dropout=0.1,
                 label_input=True, **_):
        super().__init__()
        self.conv_type, self.label_input = conv, label_input
        self.in_drop, self.drop = nn.Identity(), nn.Identity()
        self.enc = nn.Linear(in_dim + n_classes, hidden)
        self.enc_norm = nn.BatchNorm1d(hidden)
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(layers):
            if conv == "sage":
                self.convs.append(WSAGEConv(hidden, hidden))
            elif conv == "gcn":
                self.convs.append(GCNConv(hidden, hidden, add_self_loops=True, normalize=True))
            else:
                self.convs.append(GATv2Conv(hidden, hidden // heads, heads=heads, concat=True, dropout=attn_dropout,
                                            add_self_loops=True, edge_dim=1))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.head = nn.Linear(hidden * (layers + 1), n_classes)

    def _conv(self, conv, h, g):
        if self.conv_type == "sage":
            return conv(h, g["ei"], g["w"])
        if self.conv_type == "gcn":
            return conv(h, g["ei"], g["w_gcn"])
        return conv(h, g["ei"], g["ea"])

    def forward(self, x, g, label_onehot):
        if not self.label_input:
            label_onehot = torch.zeros_like(label_onehot)
        h = Fn.relu(self.enc_norm(self.enc(torch.cat([x, label_onehot], dim=1))))
        hs = [h]
        for conv, norm in zip(self.convs, self.norms):
            h = h + Fn.relu(norm(self._conv(conv, h, g)))
            hs.append(h)
        return self.head(torch.cat(hs, dim=1))


def undirected_knn(XY):
    """k = 8 nearest-neighbour graph, undirected, self excluded (as in the notebook)."""
    n = len(XY)
    _, nn_i = cKDTree(XY).query(XY, k=K + 2)
    nn_i = nn_i[nn_i != np.arange(n)[:, None]].reshape(n, K + 1)[:, :K]
    return to_undirected(torch.tensor(np.vstack([np.repeat(np.arange(n), K), nn_i.ravel()])), num_nodes=n)


def graph_tensors(edge_index, XY, dist_scale, logd_stats):
    """Edge weights of the model: softmax(-d / dist_scale) over each node's incoming edges (plain mean if None)."""
    n = len(XY)
    src, dst = edge_index
    d = torch.tensor(np.hypot(*(XY[src.numpy()] - XY[dst.numpy()]).T), dtype=torch.float32)
    deg = torch.bincount(dst, minlength=n).float()
    w = 1.0 / deg[dst] if dist_scale is None else softmax(-d / dist_scale, dst, num_nodes=n)
    ea = ((torch.log1p(d) - logd_stats[0]) / logd_stats[1]).unsqueeze(1)
    return {"ei": edge_index, "w": w, "w_gcn": w * deg[dst], "ea": ea}


# --------------------------------------------------------------------------------------------------------------------
# Features (the code of poi_feature_engineering.ipynb, applied to the existing nodes plus the new one)
# --------------------------------------------------------------------------------------------------------------------
def neighbours(XY):
    """32 nearest other nodes and their distances, self excluded explicitly (as in the feature notebook)."""
    n = len(XY)
    dk, ik = cKDTree(XY).query(XY, k=KMAX + 2)
    not_self = ik != np.arange(n)[:, None]
    assert (not_self.sum(1) == KMAX + 1).all()
    return ik[not_self].reshape(n, KMAX + 1)[:, :KMAX], dk[not_self].reshape(n, KMAX + 1)[:, :KMAX]


def knn_adjacency(nbr, n):
    src, dst = np.repeat(np.arange(n), K), nbr[:, :K].ravel()
    A_dir = sparse.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    return A_dir, ((A_dir + A_dir.T) > 0).astype(float).tocsr()


def build_towns(XY, density_2km, is_urban, adm2_name, tree):
    uidx = np.flatnonzero(is_urban)
    up = cKDTree(XY[uidx]).query_pairs(TOWN_RADIUS_M, output_type="ndarray")
    _, comp = connected_components(sparse.coo_matrix((np.ones(len(up)), (up[:, 0], up[:, 1])),
                                                     shape=(len(uidx),) * 2), directed=False)
    u = pd.DataFrame({"node": uidx, "comp": comp, "density_2km": density_2km[uidx], "adm2_name": adm2_name[uidx]})
    g = u.groupby("comp")
    towns = pd.DataFrame({"name": g["adm2_name"].agg(lambda s: s.mode()[0]), "urban_nodes": g.size(),
                          "centre_node": u.loc[g["density_2km"].idxmax(), "node"].to_numpy()})
    towns = towns.sort_values("urban_nodes", ascending=False, kind="stable")
    u["town"] = u["comp"].map(pd.Series(np.arange(len(towns)), index=towns.index))
    towns = towns.reset_index(drop=True)
    TXY = XY[towns["centre_node"].to_numpy()]
    u["d_centre_m"] = np.hypot(*(XY[u["node"]] - TXY[u["town"]]).T)
    towns["p90_urban_km"] = (u.groupby("town")["d_centre_m"].quantile(0.9) / 1000).round(2)
    towns["radius_km"] = towns["p90_urban_km"].clip(lower=TOWN_RADIUS_M / 2000)
    towns["pois_5km"] = tree.query_ball_point(TXY, 5000, return_length=True) - 1
    return towns, TXY


def neighbourhood_shape(XY, nbr, k):
    off = XY[nbr[:, :k]] - XY[:, None, :]
    dist = np.hypot(off[..., 0], off[..., 1])
    mean_dist = dist.mean(1)
    c = off - off.mean(1, keepdims=True)
    lam = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", c, c) / k)
    linearity = np.where(lam[:, 1] > 0, 1 - lam[:, 0] / np.where(lam[:, 1] > 0, lam[:, 1], 1), 0.0)
    offset = np.where(mean_dist > 0, np.hypot(*off.mean(1).T) / np.where(mean_dist > 0, mean_dist, 1), 0.0)
    ang = np.sort(np.arctan2(off[..., 1], off[..., 0]), axis=1)
    gaps = np.c_[np.diff(ang, axis=1), 2 * np.pi - (ang[:, -1] - ang[:, 0])]
    return linearity.clip(0, 1), offset.clip(0, 1), gaps.max(1) / (2 * np.pi)


def rwse(A, steps, cols, chunk=1024):
    """Random-walk return probabilities diag(P^t) for the nodes in `cols`, by sparse powers of P = D^-1 A.
    (The feature notebook uses an eigendecomposition; it checks the two agree to 1e-8.)"""
    n = A.shape[0]
    P = sparse.diags(1 / np.asarray(A.sum(1)).ravel()) @ A
    out = np.zeros((len(cols), len(steps)))
    for s in range(0, len(cols), chunk):
        c = cols[s:s + chunk]
        X = np.zeros((n, len(c)))
        X[c, np.arange(len(c))] = 1.0
        for t in range(1, max(steps) + 1):
            X = P @ X
            if t in steps:
                out[s:s + len(c), steps.index(t)] = X[c, np.arange(len(c))]
    return out.clip(0, 1)


def engineer(df, feature_names, rw_base=None, rw_cols=None):
    """Feature table for every row of df (raw EDA columns), with the columns in feature_names.
    rw_base / rw_cols: reuse stored random-walk encodings and recompute them only for the nodes in rw_cols."""
    n = len(df)
    XY = df[["x_km", "y_km"]].to_numpy() * 1000
    F = {}
    for c, (name, logged) in RAW_NAMES.items():
        F[name] = np.log1p(df[c].to_numpy(float)) if logged else df[c].to_numpy(float)

    tree = cKDTree(XY)
    nbr, nbr_d = neighbours(XY)
    for r in [100, 250, 1000, 5000, 20000]:
        label = f"{r}m" if r < 1000 else f"{r // 1000}km"
        F[f"loc_log_density_{label}"] = np.log1p(tree.query_ball_point(XY, r, return_length=True) - 1)
    for k in [3, 16, 32]:
        F[f"loc_log_knn{k}_dist_m"] = np.log1p(nbr_d[:, k - 1])
    for k in [K, KMAX]:
        lin, offs, gap = neighbourhood_shape(XY, nbr, k)
        F[f"loc_shape_linearity_k{k}"], F[f"loc_shape_offset_k{k}"], F[f"loc_shape_gap_k{k}"] = lin, offs, gap

    towns, TXY = build_towns(XY, df["density_2km"].to_numpy(), df["is_urban"].to_numpy().astype(bool),
                             df["adm2_name"].to_numpy(), tree)
    D_town = np.hypot(XY[:, None, 0] - TXY[None, :, 0], XY[:, None, 1] - TXY[None, :, 1])
    order = np.argsort(D_town, axis=1)
    i1, i2 = order[:, 0], order[:, 1]
    d1, d2 = D_town[np.arange(n), i1], D_town[np.arange(n), i2]
    size = towns["pois_5km"].to_numpy()
    F["loc_log_town_size"] = np.log1p(size[i1])
    F["loc_log_town_rel_dist"] = np.log1p(d1 / 1000 / towns["radius_km"].to_numpy()[i1])
    F["loc_log_dist_town2_km"] = np.log1p(d2 / 1000)
    for beta_km in [5, 25]:
        F[f"loc_log_town_access_{beta_km}km"] = np.log1p((size * np.exp(-D_town / (beta_km * 1000))).sum(1))

    a2, a1 = df["adm2_pcode"], df["adm1_pcode"]
    F["adm_log_adm2_nodes"] = np.log1p(a2.map(a2.value_counts())).to_numpy(float)
    F["adm_log_adm1_nodes"] = np.log1p(a1.map(a1.value_counts())).to_numpy(float)
    F["adm_adm2_urban_share"] = df.groupby("adm2_pcode")["is_urban"].transform("mean").to_numpy(float)
    F["adm_density_pct_in_adm2"] = df.groupby("adm2_pcode")["density_2km"].rank(pct=True).to_numpy(float)
    centre = pd.DataFrame(XY, columns=["x", "y"]).groupby(a2.to_numpy()).transform("mean").to_numpy()
    F["adm_log_dist_adm2_centre_km"] = np.log1p(np.hypot(*(XY - centre).T) / 1000)
    for name in feature_names:
        if name.startswith("adm_district_"):
            F[name] = a1.eq(name.removeprefix("adm_district_")).to_numpy(float)

    nb = nbr[:, :KMAX]
    share = lambda col: df[col].to_numpy(float)[nb].mean(1)
    F["ctx_named_share_k32"], F["ctx_footprint_share_k32"] = share("has_name"), share("has_footprint")
    F["ctx_point_share_k32"], F["ctx_address_share_k32"] = share("is_point"), share("has_address")
    F["ctx_hours_share_k32"] = share("has_hours")
    rank = df["osm_id_rank"].to_numpy(float)
    F["ctx_osm_id_rank_mean_k32"], F["ctx_osm_id_rank_std_k32"] = rank[nb].mean(1), rank[nb].std(1)
    F["ctx_osm_id_rank_vs_k32"] = rank - rank[nb].mean(1)
    fp, has_fp = df["log_footprint_m2"].to_numpy(float), df["has_footprint"].to_numpy().astype(bool)
    nb_fp, nb_has = fp[nb], has_fp[nb]
    n_fp = nb_has.sum(1)
    nb_fp_mean = np.where(n_fp > 0, (nb_fp * nb_has).sum(1) / np.maximum(n_fp, 1), 0.0)
    F["ctx_footprint_vs_k32"] = np.where(has_fp & (n_fp > 0), fp - nb_fp_mean, 0.0)

    A_dir, A = knn_adjacency(nbr, n)
    deg = np.asarray(A.sum(1)).ravel()
    tri = np.asarray((A @ A).multiply(A).sum(1)).ravel() / 2
    F["struct_knn8_in_degree"] = np.asarray(A_dir.sum(0)).ravel()
    F["struct_knn8_reciprocity"] = np.asarray(A_dir.multiply(A_dir.T).sum(1)).ravel() / K
    F["struct_knn8_clustering"] = 2 * tri / (deg * (deg - 1))
    rw_names = [c for c in feature_names if c.startswith("struct_rwse_")]
    steps = [int(c.split("_")[-1]) for c in rw_names]
    if rw_base is None:
        rw = rwse(A, steps, np.arange(n))
    else:
        rw = rw_base.copy()
        if len(rw_cols):
            rw[rw_cols] = rwse(A, steps, rw_cols)
    for j, c in enumerate(rw_names):
        F[c] = rw[:, j]

    missing = [c for c in feature_names if c not in F]
    assert not missing, f"features not rebuilt: {missing}"
    X = np.column_stack([F[c] for c in feature_names]).round(6)
    assert np.isfinite(X).all()
    info = {"A": A, "nbr": nbr, "town_names": towns["name"].to_numpy(), "nearest_town": i1, "dist_town_km": d1 / 1000}
    return X, info



# --------------------------------------------------------------------------------------------------------------------
# The predictor: loads the data and the 25 models once
# --------------------------------------------------------------------------------------------------------------------
class Predictor:
    def __init__(self):
        t0 = time.time()
        self.base = pd.read_csv(PROCESSED_CSV, keep_default_na=False, na_values=[""])
        self.eng = pd.read_csv(ENGINEERED_CSV, keep_default_na=False, na_values=[""])
        assert self.base["node_id"].tolist() == list(range(len(self.base))) == self.eng["node_id"].tolist()
        assert self.base["osm_id"].equals(self.eng["osm_id"])
        ck = torch.load(MODEL_PT, weights_only=False)
        self.cfg, self.feats, self.classes = ck["cfg"], ck["features"], ck["cats"]
        self.C = len(self.classes)
        self.own_idx = np.array([self.feats.index(c) for c in ck["own_cols"]])
        cont = set(ck["continuous"])
        self.cont = np.array([c in cont for c in self.feats])
        assert [c for c in self.feats if c in cont] == ck["continuous"]
        self.scaler = {}
        for f in range(N_FOLDS):
            mu, sd = np.array(ck["scalers"][f]["mean"]), np.array(ck["scalers"][f]["sd"])
            sd[sd == 0] = 1.0
            self.scaler[f] = (mu, sd)
        self.models = {f: [] for f in range(N_FOLDS)}
        for rec in ck["models"]:                          # stored in split, then seed order (as the temperatures)
            net = GNN(len(self.feats) + 1, self.C, **self.cfg)
            net.load_state_dict(rec["state"])
            self.models[int(rec["fold"])].append(net.eval())
        self.temps = {m: {int(f): list(t) for f, t in ck["temperatures"][m].items()} for m in MODES}
        assert all(len(self.temps[m][f]) == len(self.models[f]) for m in MODES for f in range(N_FOLDS))

        # Raw inputs needed to add a node: latrine positions and the stored random-walk encodings
        raw = pd.read_csv(RAW_CSV, dtype=str, encoding="utf-8-sig", keep_default_na=False, na_values=[""],
                          usecols=["amenity", "longitude", "latitude"])
        toilets = raw[raw["amenity"].str.strip().str.lower().eq("toilets")]
        self.latrine_tree = cKDTree(np.c_[TO_UTM.transform(toilets["longitude"].astype(float).to_numpy(),
                                                           toilets["latitude"].astype(float).to_numpy())])
        self.rw_names = [c for c in self.feats if c.startswith("struct_rwse_")]
        xy0 = self.base[["x_km", "y_km"]].to_numpy() * 1000
        ei0 = undirected_knn(xy0)
        logd = torch.log1p(torch.tensor(np.hypot(*(xy0[ei0[0].numpy()] - xy0[ei0[1].numpy()]).T), dtype=torch.float32))
        self.logd = (logd.mean(), logd.std())             # GAT edge-feature scaling of the notebook
        self._set_base(self.base, self.eng[self.rw_names].to_numpy(np.float64), self.eng[self.feats].to_numpy(np.float64))

        # Stored notebook predictions for the existing POIs (out-of-fold for the labelled ones)
        oof = pd.read_csv(OOF_CSV).set_index("node_id")
        unl = pd.read_csv(UNL_CSV).set_index("node_id")
        self.stored = {m: pd.concat([oof, unl])[[f"p_{m}_{c}" for c in self.classes]].sort_index() for m in MODES}
        assert self.stored["site"].index.tolist() == list(range(self.n))
        self.class_metrics = {}
        try:
            pc = json.loads(METRICS_JSON.read_text(encoding="utf-8"))["final"]["per_class"]
            self.class_metrics = {m: {r["index"]: (r["F1"], r["top-3"]) for r in pc[m]} for m in MODES}
        except (OSError, KeyError, ValueError):
            pass
        self.gaps = pd.read_csv(GAPS_CSV, keep_default_na=False) if GAPS_CSV.exists() else None
        print(f"Model loaded: {sum(len(v) for v in self.models.values())} GraphSAGE models (dual-mode), "
              f"{len(self.feats)} features, {self.n:,} existing POIs ({time.time() - t0:.1f}s)")

    def _set_base(self, base, rw_base, X0):
        """The existing graph new nodes are added to: raw EDA table, engineered features, random-walk encodings."""
        self.base, self.rw_base, self.X0, self.n = base, rw_base, X0, len(base)
        self.fold = base["fold"].to_numpy()
        self.labelled = base["is_labelled"].eq(1).to_numpy()
        self.y = base["label"].to_numpy()
        self.cat = base["category"].to_numpy()
        self.visible = {f: self.labelled & (self.fold != f) & (self.fold != (f + 1) % N_FOLDS) for f in range(N_FOLDS)}
        self.XY = base[["x_km", "y_km"]].to_numpy() * 1000
        self.tree = cKDTree(self.XY)
        self.EI = undirected_knn(self.XY)
        self.A0 = knn_adjacency(neighbours(self.XY)[0], self.n)[1]
        self.__dict__.pop("_txy0", None)

    # -- model ---------------------------------------------------------------------------------------------------------
    def _scale(self, X, f, site_rows):
        Xs = X.copy()
        mu, sd = self.scaler[f]
        Xs[:, self.cont] = (Xs[:, self.cont] - mu) / sd
        Xs = np.c_[Xs, np.zeros(len(Xs))]
        if site_rows is not None and len(site_rows):
            Xs[np.ix_(site_rows, self.own_idx)] = 0.0          # own-node attributes hidden ...
            Xs[site_rows, -1] = 1.0                            # ... and flagged as hidden
        return torch.tensor(Xs, dtype=torch.float32)

    def _score(self, X, edge_index, XY, targets, mode, folds):
        """Temperature-scaled ensemble probabilities of the target nodes for each split: {fold: (len(targets), C)}.
        Only the targets' L-hop neighbourhood is run (their output depends on nothing else). In site mode the
        targets' own attributes are hidden. Neighbour categories: the split's training labels (new nodes: none)."""
        targets = torch.as_tensor(np.atleast_1d(targets))
        g_full = graph_tensors(edge_index, XY, self.cfg["dist_scale"], self.logd)
        subset, ei_sub, mapping, emask = k_hop_subgraph(targets, max(self.cfg["layers"], 1), edge_index,
                                                        relabel_nodes=True, num_nodes=len(XY))
        g = {"ei": ei_sub, "w": g_full["w"][emask], "w_gcn": g_full["w_gcn"][emask], "ea": g_full["ea"][emask]}
        sub = subset.numpy()
        old = sub[sub < self.n]
        out = {}
        with torch.no_grad():
            for f in folds:
                x = self._scale(X[sub], f, mapping.numpy() if mode == "site" else None)
                lab = torch.zeros(len(sub), self.C)
                vis = np.flatnonzero(sub < self.n)[self.visible[f][old]]
                lab[vis, self.y[sub[vis]]] = 1.0
                out[f] = np.mean([Fn.softmax(m(x, g, lab)[mapping] / t, 1).numpy()
                                  for m, t in zip(self.models[f], self.temps[mode][f])], axis=0)
        return out

    # -- existing POIs -------------------------------------------------------------------------------------------------
    def find(self, query):
        q = str(query).strip()
        if q.isdigit() and int(q) < self.n:
            return [int(q)]
        hit = self.base.index[self.base["osm_id"].eq(q) |
                              self.base["osm_ids"].fillna("").str.split(";").apply(lambda s: q in s)]
        if len(hit):
            return hit.tolist()
        return self.base.index[self.base["name"].str.contains(q, case=False, na=False, regex=False)].tolist()

    def predict_existing(self, i):
        """The notebook's predictions: labelled POIs by the split that held their fold out, unlabelled by all 25."""
        r = self.base.loc[i]
        return {"probs": {m: self.stored[m].loc[i].to_numpy() for m in MODES},
                "true": r["category"] if self.labelled[i] else None, "lat": r["lat"], "lon": r["lon"],
                "title": f"node {i} | {r['osm_id']} | {r['name'] if isinstance(r['name'], str) else '(unnamed)'} | "
                         f"tagged {r['tag_key']}={r['tag_value']}",
                "context": self._context(i, self.base, self.base["nearest_town"].iloc[i], r["dist_town_km"],
                                         self.cat[self._knn(self.XY[i], exclude=i)]),
                "note": (f"out-of-fold predictions (models that did not train on fold {self.fold[i]})"
                         if self.labelled[i] else "unlabelled POI: mean of all 25 models"),
                "adm2_pcode": r["adm2_pcode"]}

    def _knn(self, xy, exclude=None):
        _, idx = self.tree.query(xy, k=K + 1)
        return [j for j in idx if j != exclude][:K]

    # -- new nodes -----------------------------------------------------------------------------------------------------
    def _new_row(self, lat, lon, has_name, has_address, has_hours, footprint_m2, is_point, osm_id_rank, adm2_pcode):
        lat, lon = float(lat), float(lon)
        if not (BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]):
            raise ValueError(f"({lat}, {lon}) is outside Lesotho's bounding box "
                             f"(lat {BBOX[1]} to {BBOX[3]}, lon {BBOX[0]} to {BBOX[2]})")
        footprint_m2 = float(footprint_m2 or 0)
        has_fp = int(footprint_m2 > 0)
        is_point = int(not has_fp) if is_point is None else int(is_point)
        x, y = TO_UTM.transform(lon, lat)
        xy_new = np.array([round(x / 1000, 4), round(y / 1000, 4)]) * 1000
        d = np.hypot(*(self.XY - xy_new).T)                     # distance of every existing node to the new one
        b = self.base
        if adm2_pcode and not b["adm2_pcode"].eq(adm2_pcode).any():
            raise ValueError(f"unknown adm2_pcode {adm2_pcode!r}")
        # District and sub-district: given, or those of the nearest existing POI (no boundary polygons here)
        src = b.index[b["adm2_pcode"].eq(adm2_pcode)][0] if adm2_pcode else int(np.argmin(d))
        nb_d = np.sort(d)
        new_rank = (osm_id_rank if osm_id_rank is not None
                    else float(b["osm_id_rank"].to_numpy()[np.argsort(d)[:KMAX]].mean()))
        new = {
            "x_km": xy_new[0] / 1000, "y_km": xy_new[1] / 1000,
            "density_500m": int((d <= 500).sum()), "density_2km": int((d <= 2000).sum()),
            "density_10km": int((d <= 10000).sum()), "nn1_dist_m": round(nb_d[0], 1),
            "knn8_mean_dist_m": round(nb_d[:K].mean(), 1),
            "latrines_1km": int(self.latrine_tree.query_ball_point(xy_new, 1000, return_length=True)),
            "has_name": int(has_name), "has_address": int(has_address), "has_hours": int(has_hours),
            "has_footprint": has_fp, "log_footprint_m2": round(float(np.log1p(footprint_m2)), 4), "is_point": is_point,
            "quality_score": np.mean([int(has_name), int(has_address), int(has_hours), has_fp]),
            "osm_id_rank": new_rank,
            **{c: b.loc[src, c] for c in ["adm1_pcode", "adm1_name", "adm2_pcode", "adm2_name"]},
        }
        new["is_urban"] = int(new["density_2km"] >= TOWN_MIN_POIS)
        return lat, lon, xy_new, d, new

    def predict_site(self, lat, lon, adm2_pcode=None, folds=None):
        """Site mode for a candidate location: a one-way node (it receives messages from the POIs around it and
        changes nothing for them) whose own mapping attributes are hidden."""
        lat, lon, xy_new, d, new = self._new_row(lat, lon, 0, 0, 0, 0.0, None, None, adm2_pcode)
        df = pd.concat([self.base, pd.DataFrame([new])], ignore_index=True)
        XY_all = np.vstack([self.XY, xy_new])
        _, TXY = build_towns(XY_all, df["density_2km"].to_numpy(), df["is_urban"].to_numpy().astype(bool),
                             df["adm2_name"].to_numpy(), cKDTree(XY_all))
        d_town = np.hypot(*(TXY - xy_new).T)
        df.loc[self.n, "dist_town_km"] = round(d_town.min() / 1000, 3)
        df.loc[self.n, "dist_maseru_km"] = round(d_town[0] / 1000, 3)
        rw_base = np.vstack([self.rw_base, np.zeros((1, len(self.rw_names)))])
        X_all, info = engineer(df, self.feats, rw_base=rw_base, rw_cols=np.array([self.n]))
        X = np.vstack([self.X0, X_all[self.n]])                  # existing nodes keep their features
        nbr = info["nbr"][:, :K]
        into = np.union1d(nbr[self.n], np.flatnonzero((nbr[:self.n] == self.n).any(1)))   # undirected k-NN partners
        ei = torch.cat([self.EI, torch.tensor(np.vstack([into, np.full(len(into), self.n)]))], dim=1)
        folds = list(range(N_FOLDS)) if folds is None else folds
        fp = {f: v[0] for f, v in self._score(X, ei, XY_all, [self.n], "site", folds).items()}
        town = info["town_names"][info["nearest_town"][self.n]]
        return {"probs": {"site": np.mean(list(fp.values()), axis=0)}, "fold_probs": fp, "true": None,
                "lat": lat, "lon": lon, "x": X[self.n], "adm2_pcode": new["adm2_pcode"],
                "title": f"Candidate site at ({lat:.5f}, {lon:.5f})",
                "context": self._context(self.n, df, town, info["dist_town_km"][self.n], self.cat[nbr[self.n]]),
                "note": f"site mode, mean of {sum(len(self.models[f]) for f in folds)} models",
                "town": town, "dist_town_km": float(info["dist_town_km"][self.n]), "density_2km": new["density_2km"]}

    def predict_new(self, lat, lon, has_name=0, has_address=0, has_hours=0, footprint_m2=0.0, is_point=None,
                    osm_id_rank=None, adm2_pcode=None, folds=None):
        """Known mode for a POI described as it would be mapped: fully inserted into the graph, as a newly mapped
        place would be, and every feature recomputed."""
        lat, lon, xy_new, d, new = self._new_row(lat, lon, has_name, has_address, has_hours, footprint_m2, is_point,
                                                 osm_id_rank, adm2_pcode)
        # Raw EDA columns: existing nodes keep their stored values, updated for the one extra node nearby
        df = self.base.copy()
        for r, c in [(500, "density_500m"), (2000, "density_2km"), (10000, "density_10km")]:
            df[c] = df[c] + (d <= r)
        df["is_urban"] = (df["density_2km"] >= TOWN_MIN_POIS).astype(int)
        df["nn1_dist_m"] = np.minimum(df["nn1_dist_m"], d.round(1))
        _, nb_d0 = self.tree.query(self.XY, k=K + 1)            # column 0 is the node itself (or a twin at 0 m)
        for i in np.flatnonzero(d < nb_d0[:, K]):               # the new node enters these nodes' 8 nearest
            dd = np.sort(np.r_[np.hypot(*(self.XY - self.XY[i]).T), d[i]])[1:K + 1]
            df.loc[i, "knn8_mean_dist_m"] = round(dd.mean(), 1)
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)

        # Towns: if the extra node changes them, distances to towns are recomputed for every node
        XY_all = np.vstack([self.XY, xy_new])
        towns, TXY = build_towns(XY_all, df["density_2km"].to_numpy(), df["is_urban"].to_numpy().astype(bool),
                                 df["adm2_name"].to_numpy(), cKDTree(XY_all))
        D = np.hypot(XY_all[:, None, 0] - TXY[None, :, 0], XY_all[:, None, 1] - TXY[None, :, 1])
        towns_changed = not np.array_equal(TXY, self._base_TXY())
        rows = slice(None) if towns_changed else slice(self.n, None)
        df.loc[rows, "dist_town_km"] = (D[rows].min(1) / 1000).round(3)
        df.loc[rows, "dist_maseru_km"] = (D[rows, 0] / 1000).round(3)

        # Random-walk encodings change only near the new node: recompute them within 11 hops of any changed edge
        nbr_all, _ = neighbours(XY_all)
        A_new = knn_adjacency(nbr_all, self.n + 1)[1]
        A_old = sparse.block_diag([self.A0, sparse.csr_matrix((1, 1))]).tocsr()
        changed = np.unique(np.concatenate(sparse.triu(abs(A_new - A_old)).nonzero()))
        union = ((A_new + A_old) > 0).astype(float).tocsr()
        steps = max(int(c.split("_")[-1]) for c in self.rw_names)
        dist = dijkstra(union, directed=False, indices=changed, unweighted=True, limit=steps + 0.5)
        rw_cols = np.flatnonzero(np.isfinite(np.atleast_2d(dist)).any(0))
        rw_base = np.vstack([self.rw_base, np.zeros((1, len(self.rw_names)))])
        X, info = engineer(df, self.feats, rw_base=rw_base, rw_cols=rw_cols)

        folds = list(range(N_FOLDS)) if folds is None else folds
        fp = {f: v[0] for f, v in self._score(X, undirected_knn(XY_all), XY_all, [self.n], "known", folds).items()}
        return {"probs": {"known": np.mean(list(fp.values()), axis=0)}, "fold_probs": fp, "true": None,
                "lat": lat, "lon": lon, "x": X[self.n], "adm2_pcode": new["adm2_pcode"],
                "title": f"New POI at ({lat:.5f}, {lon:.5f}) | name {bool(has_name)}, address {bool(has_address)}, "
                         f"hours {bool(has_hours)}, outline {float(footprint_m2 or 0):,.0f} m2, point {bool(new['is_point'])}",
                "context": self._context(self.n, df, info["town_names"][info["nearest_town"][self.n]],
                                         info["dist_town_km"][self.n], self.cat[nbr_all[self.n, :K]]),
                "note": f"known mode, mean of {sum(len(self.models[f]) for f in folds)} models"
                        + ("; adding it changed the town centres" if towns_changed else "")
                        + ("" if osm_id_rank is not None else "; mapping date set to that of its surroundings")}

    def _base_TXY(self):
        if not hasattr(self, "_txy0"):
            self._txy0 = build_towns(self.XY, self.base["density_2km"].to_numpy(),
                                     self.base["is_urban"].to_numpy().astype(bool), self.base["adm2_name"].to_numpy(),
                                     self.tree)[1]
        return self._txy0

    def _context(self, i, df, town, dist_town_km, nbr_cats):
        r = df.iloc[i]
        cats = pd.Series(nbr_cats).value_counts()
        return (f"{r['adm1_name']} district, {r['adm2_name']} ({r['adm2_pcode']}) | nearest town {town} "
                f"{float(dist_town_km):.1f} km | {int(r['density_500m'])} POIs within 500 m, "
                f"{int(r['density_2km'])} within 2 km | {'urban' if r['density_2km'] >= TOWN_MIN_POIS else 'rural'}\n"
                f"  8 nearest mapped POIs: " + ", ".join(f"{c} {n}" for c, n in cats.items()))

    # -- location intelligence -----------------------------------------------------------------------------------------
    def suggest(self, category, lat, lon, radius_km=2.0, step_m=None, max_points=49, top=10, max_gap_m=1500,
                progress=True):
        """Score a grid of candidate sites around a location in site mode and rank them by the probability of
        `category`. Points more than max_gap_m from any mapped POI are skipped (outside the mapped area the model
        has nothing to go on)."""
        if category not in self.classes:
            raise ValueError(f"unknown category {category!r}; choose from {self.classes}")
        c = self.classes.index(category)
        cx, cy = TO_UTM.transform(float(lon), float(lat))
        r = float(radius_km) * 1000
        step = step_m or max(100.0, np.sqrt(np.pi * r * r / max_points))
        g = np.arange(-r, r + 1e-6, step)
        pts = np.array([(cx + dx, cy + dy) for dx in g for dy in g if dx * dx + dy * dy <= r * r])
        near, _ = self.tree.query(pts)
        pts = pts[near <= max_gap_m]
        if not len(pts):
            raise ValueError(f"no mapped POI within {max_gap_m} m of any grid point; try a larger radius")
        rows = []
        t0 = time.time()
        for k, (x, y) in enumerate(pts):
            plon, plat = TO_LL.transform(x, y)
            if not (BBOX[0] <= plon <= BBOX[2] and BBOX[1] <= plat <= BBOX[3]):
                continue
            res = self.predict_site(plat, plon)
            p = res["probs"]["site"]
            within = self.tree.query_ball_point([x, y], 1000)
            rows.append({"lat": round(plat, 5), "lon": round(plon, 5), f"p {category}": p[c],
                         "rank": int((p > p[c]).sum()) + 1, "most likely": self.classes[int(p.argmax())],
                         f"{category} within 1 km": int((self.cat[within] == category).sum()),
                         "POIs within 1 km": len(within), "town": res["town"], "town km": round(res["dist_town_km"], 1),
                         "sub-district": res["adm2_pcode"]})
            if progress:
                print(f"\r  scored {k + 1}/{len(pts)} candidate sites ({time.time() - t0:.0f}s)", end="", flush=True)
        if progress:
            print()
        out = pd.DataFrame(rows).sort_values(f"p {category}", ascending=False).reset_index(drop=True)
        out.index += 1
        return out.head(top), len(out), step

    def gap_rows(self, area=None, adm2_pcode=None, only_flagged=False):
        if self.gaps is None:
            raise ValueError(f"{GAPS_CSV.name} not found: run gnn_location_model.ipynb (Section 9) first")
        g = self.gaps
        if adm2_pcode:
            g = g[g["adm2_pcode"].eq(adm2_pcode)]
        if area:
            a = str(area).lower()
            g = g[g["adm2_name"].str.lower().str.contains(a, regex=False) |
                  g["adm1_name"].str.lower().str.contains(a, regex=False) | g["adm2_pcode"].str.lower().eq(a)]
        if only_flagged:
            g = g[g["signal"].ne("")]
        return g

    # -- checks --------------------------------------------------------------------------------------------------------
    def check(self, n_reinsert=3, n_site=30):
        print(f"Check 1: rebuilding all {len(self.feats)} features of the {self.n:,} existing POIs ...")
        X, _ = engineer(self.base, self.feats)
        diff = np.abs(X - self.X0)
        worst = pd.Series(diff.max(0), index=self.feats).sort_values(ascending=False)
        print(f"  largest difference from poi_engineered.csv: {worst.iloc[0]:.2e} ({worst.index[0]})")
        assert worst.iloc[0] < 5e-6, worst.head()

        print("Check 2: live predictions for every held-out fold against model_outputs_v2/oof_predictions.csv ...")
        live = {m: np.full((self.n, self.C), np.nan) for m in MODES}
        rng = np.random.default_rng(0)
        for f in range(N_FOLDS):
            te = np.flatnonzero(self.labelled & (self.fold == f))
            live["known"][te] = self._score(self.X0, self.EI, self.XY, te, "known", [f])[f]
            for grp in np.array_split(rng.permutation(te), 10):      # site mode: test nodes masked in 10 groups
                live["site"][grp] = self._score(self.X0, self.EI, self.XY, grp, "site", [f])[f]
        lab = np.flatnonzero(self.labelled)
        for m in MODES:
            st = self.stored[m].to_numpy()[lab]
            lv = live[m][lab]
            agree = (st.argmax(1) == lv.argmax(1)).mean()
            f1 = lambda p: f1_score(self.y[lab], p.argmax(1), average="macro")
            print(f"  {m:5s} mode: same predicted class for {agree:.1%} of nodes | mean probability difference "
                  f"{np.abs(st - lv).mean():.3f} | macro-F1 live {f1(lv):.3f} vs notebook {f1(st):.3f}")
            assert agree > 0.9 and abs(f1(lv) - f1(st)) < 0.02
        print("  (the notebook adds label reuse and masks 10% of the other held-out nodes at once; both are small)")

        print(f"Check 3: remove {n_reinsert} POIs one at a time and add each back as a new POI (known mode) ...")
        for i in rng.choice(lab, n_reinsert, replace=False):
            r = self.base.loc[i]
            res = self._without(int(i)).predict_new(r["lat"], r["lon"], r["has_name"], r["has_address"],
                                                    r["has_hours"], np.expm1(r["log_footprint_m2"]), r["is_point"],
                                                    r["osm_id_rank"], r["adm2_pcode"], folds=[int(r["fold"])])
            dx = np.abs(res["x"] - self.X0[i])
            dp = np.abs(res["probs"]["known"] - live["known"][i]).max()
            print(f"  node {i:4d} ({r['category']:15s}) largest feature difference {dx.max():.1e} "
                  f"({self.feats[int(dx.argmax())]}) | largest probability difference {dp:.1e}")
            assert dx.max() < 0.05 and dp < 1e-2      # 0.05: 0.1 m coordinate rounding at a 1-2 m spacing (log scale)

        print(f"Check 4: remove {n_site} POIs one at a time and score each spot as a candidate site (site mode, "
              "the models that never trained on it) ...")
        rows = []
        for i in rng.choice(lab, n_site, replace=False):
            r = self.base.loc[i]
            p = self._without(int(i)).predict_site(r["lat"], r["lon"], r["adm2_pcode"], folds=[int(r["fold"])])
            p = p["probs"]["site"]
            st = self.stored["site"].loc[i].to_numpy()
            rows.append({"same class as notebook": p.argmax() == st.argmax(), "top-3 hit": self.y[i] in np.argsort(-p)[:3],
                         "notebook top-3 hit": self.y[i] in np.argsort(-st)[:3], "diff": np.abs(p - st).max()})
        rs = pd.DataFrame(rows)
        print(f"  same predicted class as the notebook's site mode for {rs['same class as notebook'].mean():.0%} | "
              f"true category in the top 3: {rs['top-3 hit'].mean():.0%} (notebook {rs['notebook top-3 hit'].mean():.0%})"
              f" | median largest probability difference {rs['diff'].median():.3f}")
        assert rs["same class as notebook"].mean() >= 0.7
        print("All checks passed.")

    def _without(self, i):
        """A copy of the predictor whose existing graph lacks node i (for the reinsertion checks)."""
        sub = copy.copy(self)
        keep = np.arange(self.n) != i
        b = self.base[keep].reset_index(drop=True)
        XYk = self.XY[keep]
        d = np.hypot(*(XYk - self.XY[i]).T)
        for r, c in [(500, "density_500m"), (2000, "density_2km"), (10000, "density_10km")]:
            b[c] = b[c] - (d <= r)
        b["is_urban"] = (b["density_2km"] >= TOWN_MIN_POIS).astype(int)
        _, nb_d0 = self.tree.query(self.XY, k=K + 1)
        hit = np.flatnonzero(d <= nb_d0[keep][:, K])              # node i was among these nodes' 8 nearest
        dd, _ = cKDTree(XYk).query(XYk[hit], k=K + 1)
        b.loc[hit, "nn1_dist_m"] = dd[:, 1].round(1)
        b.loc[hit, "knn8_mean_dist_m"] = dd[:, 1:].mean(1).round(1)
        X, _ = engineer(b, self.feats)
        sub._set_base(b, X[:, [self.feats.index(c) for c in self.rw_names]], X)
        return sub


# --------------------------------------------------------------------------------------------------------------------
# Output and command line
# --------------------------------------------------------------------------------------------------------------------
def show(pred, res, top=8):
    print("\n" + res["title"])
    print("  " + res["context"])
    for mode, p in res["probs"].items():
        cm = pred.class_metrics.get(mode, {})
        print(f"\n  {mode.upper()} MODE" + ("  (from the location and its neighbourhood only)" if mode == "site"
                                           else "  (also uses how the POI itself is mapped)"))
        print(f"  {'category':16s} {'prob':>6s}  {'':40s} {'test F1':>7s} {'top-3':>6s}")
        order = np.argsort(p)[::-1][:top]
        for j in order:
            f1, t3 = cm.get(pred.classes[j], (np.nan, np.nan))
            mark = "  <- true category" if res["true"] == pred.classes[j] else ""
            print(f"  {pred.classes[j]:16s} {p[j]:6.1%}  {'#' * int(round(p[j] * 40)):40s} {f1:7.2f} {t3:6.2f}{mark}")
        print(f"  Most likely: " + ", ".join(f"{pred.classes[j]} ({p[j]:.0%})" for j in order[:3]))
    print(f"  ({res['note']}; 'test F1' and 'top-3' are the model's out-of-fold scores for that class in that mode.\n"
          "   Training weights the small categories up, so their probabilities run high: read the ranking, not the "
          "percentages, as odds.)")
    if pred.gaps is not None and res.get("adm2_pcode"):
        g = pred.gap_rows(adm2_pcode=res["adm2_pcode"], only_flagged=True)
        if len(g):
            print(f"\n  Market signals in this sub-district ({res['adm2_pcode']}), from {GAPS_CSV.name}:")
            print_gaps(g, indent="  ")


def print_gaps(g, indent=""):
    cols = ["adm1_name", "adm2_name", "adm2_pcode", "category", "pois", "observed", "expected", "gap", "z", "signal",
            "mapping"]
    txt = g[[c for c in cols if c in g.columns]].to_string(index=False, float_format=lambda v: f"{v:.1f}")
    print("\n".join(indent + line for line in txt.splitlines()))


def ask(prompt, default=None, cast=str):
    while True:
        s = input(f"{prompt}{f' [{default}]' if default is not None else ''}: ").strip()
        if not s and default is not None:
            return default
        try:
            return cast(s)
        except ValueError:
            print("  please enter a valid value")


def yes(prompt, default="n"):
    return ask(f"{prompt} (y/n)", default).lower().startswith("y")


def show_suggest(pred, category, lat, lon, radius_km, step_m, top):
    print(f"\nBest candidate sites for {category} within {radius_km} km of ({lat}, {lon}), site mode:")
    tab, n, step = pred.suggest(category, lat, lon, radius_km, step_m, top=top)
    print(tab.to_string(float_format=lambda v: f"{v:.3f}"))
    f1, t3 = pred.class_metrics.get("site", {}).get(category, (np.nan, np.nan))
    print(f"  {n} grid points {step:.0f} m apart were scored. 'p {category}' is how strongly the location and its "
          f"surroundings resemble where {category} POIs are mapped (site-mode test F1 for {category} {f1:.2f}, top-3 "
          f"{t3:.2f}).\n  A high probability with few {category} POIs within 1 km is a candidate gap; a high "
          "probability with many is an established cluster. Presence on the map is not proof of profitability.")


def batch(pred, path, out, known):
    src = pd.read_csv(path)
    cols = {c.lower(): c for c in src.columns}
    if "lat" not in cols or "lon" not in cols:
        raise ValueError("the CSV needs 'lat' and 'lon' columns")
    get = lambda r, c, d: r[cols[c]] if c in cols and pd.notna(r[cols[c]]) else d
    rows = []
    for k, r in src.iterrows():
        if known:
            res = pred.predict_new(get(r, "lat", None), get(r, "lon", None), int(get(r, "has_name", 0)),
                                   int(get(r, "has_address", 0)), int(get(r, "has_hours", 0)),
                                   float(get(r, "footprint_m2", 0)),
                                   None if get(r, "is_point", None) is None else int(get(r, "is_point", None)),
                                   None if get(r, "osm_id_rank", None) is None else float(get(r, "osm_id_rank", None)),
                                   get(r, "adm2_pcode", None))
        else:
            res = pred.predict_site(get(r, "lat", None), get(r, "lon", None), get(r, "adm2_pcode", None))
        mode, p = next(iter(res["probs"].items()))
        top3 = np.argsort(-p)[:3]
        rows.append({"mode": mode, "pred_category": pred.classes[top3[0]], "pred_prob": round(float(p.max()), 4),
                     "top3": "; ".join(pred.classes[j] for j in top3),
                     **{f"p_{c}": round(float(v), 4) for c, v in zip(pred.classes, p)}})
        print(f"  row {k + 1}/{len(src)}: {rows[-1]['pred_category']} ({rows[-1]['pred_prob']:.0%})")
    result = pd.concat([src, pd.DataFrame(rows)], axis=1)
    result.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {out} ({len(result)} rows)")


def interactive(pred):
    print("\nWhat do you want to do?")
    while True:
        print("\n  1  What does this location suit? (candidate site: latitude and longitude only)\n"
              "  2  Where should a given business type go? (best spots for a category around a location)\n"
              "  3  Market gaps of an area (sub-district or district name, or code such as LSA12)\n"
              "  4  A new POI described as it would be mapped (name, address, hours, outline)\n"
              "  5  An existing POI (node_id, OSM id such as node/123, or part of its name)\n"
              "  6  A CSV of candidate sites (columns lat, lon; optional adm2_pcode)\n"
              "  q  Quit")
        try:
            choice = ask("Choice", "1").lower()
            if choice == "q":
                return
            if choice == "1":
                lat = ask("  Latitude (e.g. -29.3151)", cast=float)
                lon = ask("  Longitude (e.g. 27.4869)", cast=float)
                show(pred, pred.predict_site(lat, lon))
            elif choice == "2":
                print("  Categories: " + ", ".join(pred.classes))
                cat = ask("  Category", "Retail")
                lat = ask("  Latitude of the centre (e.g. -29.3151)", cast=float)
                lon = ask("  Longitude of the centre (e.g. 27.4869)", cast=float)
                show_suggest(pred, cat, lat, lon, ask("  Radius in km", 2.0, float), None, ask("  How many", 10, int))
            elif choice == "3":
                g = pred.gap_rows(ask("  Area"), only_flagged=yes("  Only the flagged gaps and clusters?", "y"))
                print_gaps(g) if len(g) else print("  nothing found")
            elif choice == "4":
                lat = ask("  Latitude (e.g. -29.3151)", cast=float)
                lon = ask("  Longitude (e.g. 27.4869)", cast=float)
                has_name = yes("  Has a name?", "y")
                has_address = yes("  Has an address?")
                has_hours = yes("  Has opening hours?")
                fp = ask("  Building outline area in m2 (0 = mapped as a point only)", 0.0, float)
                show(pred, pred.predict_new(lat, lon, has_name, has_address, has_hours, fp))
            elif choice == "5":
                hits = pred.find(ask("  node_id, OSM id or name"))
                if not hits:
                    print("  no POI found")
                    continue
                if len(hits) > 1:
                    for j, i in enumerate(hits[:15]):
                        r = pred.base.loc[i]
                        print(f"   {j + 1:2d}. node {i} {r['osm_id']:22s} {r['name']} ({r['category']}, {r['adm2_name']})")
                    sel = ask("  Which one", 1, int)
                    hits = [hits[min(max(sel, 1), min(len(hits), 15)) - 1]]
                show(pred, pred.predict_existing(hits[0]))
            elif choice == "6":
                path = ask("  Input CSV path")
                batch(pred, path, ask("  Output CSV path", str(Path(path).with_name(Path(path).stem + "_predictions.csv"))),
                      known=False)
            else:
                print(f"  unknown choice {choice!r}")
        except (ValueError, OSError) as e:
            print(f"  error: {e}")
        except (EOFError, KeyboardInterrupt):
            print()
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify features and predictions against the notebook")
    sub = ap.add_subparsers(dest="mode")
    s = sub.add_parser("site", help="what a candidate location suits (site mode)")
    s.add_argument("--lat", type=float, required=True)
    s.add_argument("--lon", type=float, required=True)
    s.add_argument("--adm2-pcode", help="sub-district code, e.g. LSA12; default: that of the nearest POI")
    g = sub.add_parser("suggest", help="best candidate sites for a category around a location (site mode)")
    g.add_argument("--category", required=True)
    g.add_argument("--lat", type=float, required=True)
    g.add_argument("--lon", type=float, required=True)
    g.add_argument("--radius-km", type=float, default=2.0)
    g.add_argument("--step-m", type=float, help="grid spacing; default: about 49 points in the circle")
    g.add_argument("--top", type=int, default=10)
    m = sub.add_parser("gaps", help="market gaps and clusters per sub-district (from the notebook)")
    m.add_argument("--area", help="part of a sub-district or district name, or a code such as LSA12")
    m.add_argument("--all", action="store_true", help="show every category, not only the flagged ones")
    n = sub.add_parser("new", help="a new POI described as it would be mapped (known mode)")
    n.add_argument("--lat", type=float, required=True)
    n.add_argument("--lon", type=float, required=True)
    n.add_argument("--name", action="store_true", help="the POI has a name")
    n.add_argument("--address", action="store_true", help="the POI has an address")
    n.add_argument("--hours", action="store_true", help="the POI has opening hours")
    n.add_argument("--footprint-m2", type=float, default=0.0, help="building outline area; 0 = point only")
    n.add_argument("--also-point", action="store_true", help="mapped as a point as well as an outline")
    n.add_argument("--osm-id-rank", type=float, help="0-1 mapping-date rank; default: that of its surroundings")
    n.add_argument("--adm2-pcode", help="sub-district code, e.g. LSA12; default: that of the nearest POI")
    e = sub.add_parser("existing", help="the notebook's predictions for an existing POI (both modes)")
    e.add_argument("--query", required=True, help="node_id, OSM id (node/123, way/456) or part of a name")
    b = sub.add_parser("batch", help="score every row of a CSV (candidate sites, or new POIs with --known)")
    b.add_argument("csv")
    b.add_argument("--known", action="store_true", help="rows describe mapped POIs (has_name, ...): known mode")
    b.add_argument("--out", help="output CSV (default: <input>_predictions.csv)")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    pred = Predictor()
    try:
        run(pred, args)
    except (ValueError, OSError) as e:
        sys.exit(f"Error: {e}")


def run(pred, args):
    if args.check:
        pred.check()
    elif args.mode == "site":
        show(pred, pred.predict_site(args.lat, args.lon, args.adm2_pcode))
    elif args.mode == "suggest":
        show_suggest(pred, args.category, args.lat, args.lon, args.radius_km, args.step_m, args.top)
    elif args.mode == "gaps":
        g = pred.gap_rows(args.area, only_flagged=not args.all)
        print_gaps(g) if len(g) else print("Nothing found.")
    elif args.mode == "new":
        is_point = 1 if (args.also_point or args.footprint_m2 <= 0) else 0
        show(pred, pred.predict_new(args.lat, args.lon, args.name, args.address, args.hours, args.footprint_m2,
                                    is_point, args.osm_id_rank, args.adm2_pcode))
    elif args.mode == "existing":
        hits = pred.find(args.query)
        if not hits:
            sys.exit(f"No POI matches {args.query!r}")
        if len(hits) > 1:
            print(f"{len(hits)} POIs match; showing the first. Use a node_id or OSM id to pick another.")
        show(pred, pred.predict_existing(hits[0]))
    elif args.mode == "batch":
        batch(pred, args.csv, args.out or str(Path(args.csv).with_name(Path(args.csv).stem + "_predictions.csv")),
              args.known)
    else:
        interactive(pred)


if __name__ == "__main__":
    main()
