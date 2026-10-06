"""Predict the category of a point of interest (POI) in Lesotho with the trained GraphSAGE model from model.ipynb.

Run it with no arguments for an interactive session: the model is loaded once and you choose what to predict.

    python predict_poi.py                       interactive menu
    python predict_poi.py new --lat -29.315 --lon 27.487 --name --hours
    python predict_poi.py existing --query "node/4340647091"      (node_id, osm_id or part of a name)
    python predict_poi.py batch my_pois.csv --out my_predictions.csv
    python predict_poi.py --check               verify features and predictions against the notebook's outputs

How a new POI is predicted. The model is a graph model, so a new POI cannot be scored on its own: it is added to the
graph of the 6,083 existing POIs as one more node. Every feature is then recomputed the way poi_eda.ipynb and
poi_feature_engineering.ipynb compute it (densities, towns, neighbourhood shape, district context, the mapping
profile of the area and the graph-structure encodings), the k = 8 nearest-neighbour graph is rebuilt, and all 25
trained models (5 CV splits x 5 seeds) score it. The new POI's label is never shown to the models. Each POI in a batch
is added and scored on its own.

What you describe for a new POI is how it would be mapped in OpenStreetMap: whether it has a name, an address, opening
hours, a building outline (and its area), and whether it is a point. These attributes carry most of the model's
signal (model.ipynb, Section 6), so set them as the place would really be mapped.
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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="IProgress not found")
warnings.filterwarnings("ignore", message=".*torch.jit.script.*")
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from pyproj import Transformer
from torch_geometric.nn import GATv2Conv, SAGEConv
from torch_geometric.utils import to_undirected

HERE = Path(__file__).resolve().parent
PROCESSED_CSV = HERE / "processed_poi.csv"
ENGINEERED_CSV = HERE / "poi_engineered.csv"
RAW_CSV = HERE / "points_of_interest.csv"
MODEL_PT = HERE / "model_outputs" / "gnn_final_models.pt"
METRICS_JSON = HERE / "model_outputs" / "metrics.json"

K, KMAX = 8, 32                              # k-NN graph of the model; neighbourhood size of the ctx_ features
TOWN_MIN_POIS, TOWN_RADIUS_M = 50, 2000      # EDA town rule
N_FOLDS = 5
BBOX = (27.0115, -30.6756, 29.4557, -28.5706)   # Lesotho (lon_min, lat_min, lon_max, lat_max), from README.txt
TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32735", always_xy=True)   # UTM 35S, as in poi_eda.ipynb

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
# Model (identical to model.ipynb, Section 2; dropout is inactive at prediction time)
# --------------------------------------------------------------------------------------------------------------------
class GNN(nn.Module):
    def __init__(self, in_dim, n_classes, conv="sage", hidden=128, layers=2, heads=4, dropout=0.3,
                 input_dropout=0.1, attn_dropout=0.1, label_input=False, **_):
        super().__init__()
        self.label_input = label_input
        self.in_drop, self.drop = nn.Dropout(input_dropout), nn.Dropout(dropout)
        self.enc = nn.Linear(in_dim + (n_classes if label_input else 0), hidden)
        self.enc_norm = nn.BatchNorm1d(hidden)
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(layers):
            if conv == "sage":
                self.convs.append(SAGEConv(hidden, hidden, aggr="mean", root_weight=True))
            else:
                self.convs.append(GATv2Conv(hidden, hidden // heads, heads=heads, concat=True,
                                            dropout=attn_dropout, add_self_loops=True))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.head = nn.Linear(hidden * (layers + 1), n_classes)

    def forward(self, x, edge_index, label_onehot=None):
        if self.label_input:
            x = torch.cat([x, label_onehot], dim=1)
        h = self.drop(Fn.relu(self.enc_norm(self.enc(self.in_drop(x)))))
        hs = [h]
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.drop(Fn.relu(norm(conv(h, edge_index))))
            hs.append(h)
        return self.head(torch.cat(hs, dim=1))


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
        ck = torch.load(MODEL_PT, weights_only=True)
        self.cfg, self.feats, self.classes = ck["config"], ck["features"], ck["classes"]
        self.offsets = {int(f): np.array(v) for f, v in ck["offsets"].items()}
        self.C = len(self.classes)

        # Per-split scaler: continuous features only, fitted on the non-test folds (as in model.ipynb)
        binary = set(ck["binary_features"])
        self.cont = np.array([c not in binary for c in self.feats])
        X0 = self.eng[self.feats].to_numpy(np.float64)
        fold0 = self.eng["fold"].to_numpy()
        self.scaler = {}
        for f in range(N_FOLDS):
            mu, sd = X0[fold0 != f][:, self.cont].mean(0), X0[fold0 != f][:, self.cont].std(0)
            sd[sd == 0] = 1.0
            self.scaler[f] = (mu, sd)
        self.models = {f: [] for f in range(N_FOLDS)}
        for m in ck["models"]:
            net = GNN(len(self.feats), self.C, **self.cfg)
            net.load_state_dict(m["state_dict"])
            self.models[int(m["test_fold"])].append(net.eval())

        # Raw inputs needed to add a node: latrine positions and the stored random-walk encodings
        raw = pd.read_csv(RAW_CSV, dtype=str, encoding="utf-8-sig", keep_default_na=False, na_values=[""],
                          usecols=["amenity", "longitude", "latitude"])
        toilets = raw[raw["amenity"].str.strip().str.lower().eq("toilets")]
        self.latrine_tree = cKDTree(np.c_[TO_UTM.transform(toilets["longitude"].astype(float).to_numpy(),
                                                           toilets["latitude"].astype(float).to_numpy())])
        self.rw_names = [c for c in self.feats if c.startswith("struct_rwse_")]
        self._set_base(self.base, self.eng[self.rw_names].to_numpy(np.float64))

        # Predictions for the existing graph (used for existing POIs)
        self.base_probs = self._run(X0, self.XY)
        try:
            pc = json.loads(METRICS_JSON.read_text(encoding="utf-8"))["final_per_class"]
            self.class_f1 = {r["category"]: r["F1 raw"] for r in pc}
        except (OSError, KeyError, ValueError):
            self.class_f1 = {}
        print(f"Model loaded: {sum(len(v) for v in self.models.values())} GraphSAGE models, {len(self.feats)} features, "
              f"{self.n:,} existing POIs ({time.time() - t0:.1f}s)")

    def _set_base(self, base, rw_base):
        """The existing graph a new POI is added to: raw EDA table and stored random-walk encodings."""
        self.base, self.rw_base, self.n = base, rw_base, len(base)
        self.fold = base["fold"].to_numpy()
        self.labelled = base["is_labelled"].eq(1).to_numpy()
        self.y = base["label"].to_numpy()
        self.visible = {f: self.labelled & (self.fold != f) & (self.fold != (f + 1) % N_FOLDS) for f in range(N_FOLDS)}
        self.XY = base[["x_km", "y_km"]].to_numpy() * 1000
        self.tree = cKDTree(self.XY)
        self.A0 = knn_adjacency(neighbours(self.XY)[0], self.n)[1]
        self.__dict__.pop("_txy0", None)

    # -- model ---------------------------------------------------------------------------------------------------------
    def _run(self, X, XY):
        """Softmax of every node for each test-fold ensemble: {fold: (n, C)}, mean over that fold's 5 seeds."""
        n = len(XY)
        _, nn_i = cKDTree(XY).query(XY, k=K + 2)
        nn_i = nn_i[nn_i != np.arange(n)[:, None]].reshape(n, K + 1)[:, :K]
        edge_index = to_undirected(torch.tensor(np.vstack([np.repeat(np.arange(n), K), nn_i.ravel()])), num_nodes=n)
        out = {}
        with torch.no_grad():
            for f in range(N_FOLDS):
                Xs = X.copy()
                mu, sd = self.scaler[f]
                Xs[:, self.cont] = (Xs[:, self.cont] - mu) / sd
                xt = torch.tensor(Xs, dtype=torch.float32)
                lab = torch.zeros(n, self.C)
                vis = np.flatnonzero(self.visible[f])            # new nodes (index >= self.n) are never visible
                lab[vis, self.y[vis]] = 1.0
                out[f] = np.mean([Fn.softmax(m(xt, edge_index, lab), 1).numpy() for m in self.models[f]], axis=0)
        return out

    def _combine(self, probs_by_fold, folds, calibrated):
        if calibrated:
            lp = np.mean([np.log(probs_by_fold[f] + 1e-9) + self.offsets[f] for f in folds], axis=0)
            p = np.exp(lp - lp.max())
            return p / p.sum()
        return np.mean([probs_by_fold[f] for f in folds], axis=0)

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

    def predict_existing(self, i, calibrated=False):
        """Labelled POIs: the ensemble whose split held their fold out (never trained on them). Unlabelled: all 25."""
        folds = [int(self.fold[i])] if self.labelled[i] else list(range(N_FOLDS))
        p = self._combine({f: v[i] for f, v in self.base_probs.items()}, folds, calibrated)
        r = self.base.loc[i]
        return {"probs": p, "true": r["category"] if self.labelled[i] else None, "lat": r["lat"], "lon": r["lon"],
                "title": f"node {i} | {r['osm_id']} | {r['name'] if isinstance(r['name'], str) else '(unnamed)'} | "
                         f"tagged {r['tag_key']}={r['tag_value']}",
                "context": self._context(i, self.base, self.base["nearest_town"].iloc[i], r["dist_town_km"],
                                         self.base_nbr_labels(i)),
                "note": (f"out-of-fold prediction (models that did not train on fold {self.fold[i]})"
                         if self.labelled[i] else "unlabelled POI: mean of all 25 models")}

    def base_nbr_labels(self, i):
        _, idx = self.tree.query(self.XY[i], k=K + 1)
        idx = [j for j in idx if j != i][:K]
        return self.base["category"].to_numpy()[idx]

    # -- new POIs ------------------------------------------------------------------------------------------------------
    def predict_new(self, lat, lon, has_name=0, has_address=0, has_hours=0, footprint_m2=0.0, is_point=None,
                    osm_id_rank=None, adm2_pcode=None, calibrated=False):
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

        # District and sub-district: given, or those of the nearest existing POI (no boundary polygons here)
        b = self.base
        if adm2_pcode and not b["adm2_pcode"].eq(adm2_pcode).any():
            raise ValueError(f"unknown adm2_pcode {adm2_pcode!r}")
        src = b.index[b["adm2_pcode"].eq(adm2_pcode)][0] if adm2_pcode else int(np.argmin(d))

        # Raw EDA columns: existing nodes keep their stored values, updated for the one extra node nearby
        df = b.copy()
        for r, c in [(500, "density_500m"), (2000, "density_2km"), (10000, "density_10km")]:
            df[c] = df[c] + (d <= r)
        df["is_urban"] = (df["density_2km"] >= TOWN_MIN_POIS).astype(int)
        df["nn1_dist_m"] = np.minimum(df["nn1_dist_m"], d.round(1))
        _, nb_d0 = self.tree.query(self.XY, k=K + 1)            # column 0 is the node itself (or a twin at 0 m)
        hit = np.flatnonzero(d < nb_d0[:, K])                   # the new node enters these nodes' 8 nearest
        for i in hit:
            dd = np.sort(np.r_[np.hypot(*(self.XY - self.XY[i]).T), d[i]])[1:K + 1]
            df.loc[i, "knn8_mean_dist_m"] = round(dd.mean(), 1)

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

        probs = self._run(X, XY_all)
        p = self._combine({f: v[self.n] for f, v in probs.items()}, list(range(N_FOLDS)), calibrated)
        nbr_cats = self.base["category"].to_numpy()[nbr_all[self.n, :K]]
        return {"probs": p, "true": None, "lat": lat, "lon": lon, "x": X[self.n],
                "fold_probs": {f: v[self.n] for f, v in probs.items()},
                "title": f"new POI at ({lat:.5f}, {lon:.5f}) | name {bool(has_name)}, address {bool(has_address)}, "
                         f"hours {bool(has_hours)}, outline {footprint_m2:,.0f} m2, point {bool(is_point)}",
                "context": self._context(self.n, df, info["town_names"][info["nearest_town"][self.n]],
                                         info["dist_town_km"][self.n], nbr_cats),
                "note": "mean of all 25 models" + ("; adding it changed the town centres" if towns_changed else "")
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

    # -- checks --------------------------------------------------------------------------------------------------------
    def check(self, n_reinsert=5):
        print("Check 1: rebuilding all 66 features of the 6,083 existing POIs ...")
        X, _ = engineer(self.base, self.feats)
        diff = np.abs(X - self.eng[self.feats].to_numpy(np.float64))
        worst = pd.Series(diff.max(0), index=self.feats).sort_values(ascending=False)
        print(f"  largest difference from poi_engineered.csv: {worst.iloc[0]:.2e} ({worst.index[0]})")
        assert worst.iloc[0] < 5e-6, worst.head()
        print("Check 2: out-of-fold predictions against model_outputs/oof_predictions.csv ...")
        oof = pd.read_csv(HERE / "model_outputs" / "oof_predictions.csv")
        P = np.array([self.predict_existing(i)["probs"] for i in oof["node_id"]])
        d = np.abs(P - oof[[f"p_{c}" for c in self.classes]].to_numpy()).max()
        agree = (np.array(self.classes)[P.argmax(1)] == oof["pred_raw"].to_numpy()).mean()
        print(f"  largest probability difference {d:.1e} | same predicted class for {agree:.2%} of nodes")
        assert d < 1e-3 and agree > 0.999
        print(f"Check 3: remove {n_reinsert} POIs one at a time and add each back as a new POI ...")
        oof_p = oof.set_index("node_id")[[f"p_{c}" for c in self.classes]]
        rng = np.random.default_rng(0)
        for i in rng.choice(np.flatnonzero(self.labelled), n_reinsert, replace=False):
            r = self.base.loc[i]
            sub = self._without(int(i))
            res = sub.predict_new(r["lat"], r["lon"], r["has_name"], r["has_address"], r["has_hours"],
                                  np.expm1(r["log_footprint_m2"]), r["is_point"], r["osm_id_rank"], r["adm2_pcode"])
            dx = np.abs(res["x"] - self.eng.loc[i, self.feats].to_numpy(np.float64))
            p_new = res["fold_probs"][int(r["fold"])]          # the models that never trained on this POI
            dp = np.abs(p_new - oof_p.loc[i].to_numpy()).max()
            print(f"  node {i:4d} ({r['category']:15s}) largest feature difference {dx.max():.1e} "
                  f"({self.feats[int(dx.argmax())]}) | largest probability difference {dp:.1e} | "
                  f"class {self.classes[p_new.argmax()]} vs out-of-fold {self.classes[oof_p.loc[i].to_numpy().argmax()]}")
            assert dx.max() < 0.05 and dp < 1e-2      # 0.05: 0.1 m coordinate rounding at a 1-2 m spacing (log scale)
        print("All checks passed.")

    def _without(self, i):
        """A copy of the predictor whose existing graph lacks node i (for the reinsertion check)."""
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
        sub._set_base(b, X[:, [self.feats.index(c) for c in self.rw_names]])
        return sub


# --------------------------------------------------------------------------------------------------------------------
# Output and command line
# --------------------------------------------------------------------------------------------------------------------
def show(res, classes, class_f1, top=8):
    p = res["probs"]
    print("\n" + res["title"])
    print("  " + res["context"])
    order = np.argsort(p)[::-1][:top]
    for j in order:
        f1 = f"   (model test F1 for this class {class_f1[classes[j]]:.2f})" if classes[j] in class_f1 else ""
        mark = "  <- true category" if res["true"] == classes[j] else ""
        print(f"  {classes[j]:16s} {p[j]:6.1%}  {'#' * int(round(p[j] * 40)):40s}{f1 if j == order[0] else ''}{mark}")
    print(f"  Predicted: {classes[order[0]]}" + (f" | true: {res['true']}" if res["true"] else "") + f" | {res['note']}")


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


def batch(pred, path, out, calibrated):
    src = pd.read_csv(path)
    cols = {c.lower(): c for c in src.columns}
    if "lat" not in cols or "lon" not in cols:
        raise ValueError("the CSV needs 'lat' and 'lon' columns")
    get = lambda r, c, d: r[cols[c]] if c in cols and pd.notna(r[cols[c]]) else d
    rows = []
    for k, r in src.iterrows():
        res = pred.predict_new(get(r, "lat", None), get(r, "lon", None), int(get(r, "has_name", 0)),
                               int(get(r, "has_address", 0)), int(get(r, "has_hours", 0)),
                               float(get(r, "footprint_m2", 0)),
                               None if get(r, "is_point", None) is None else int(get(r, "is_point", None)),
                               None if get(r, "osm_id_rank", None) is None else float(get(r, "osm_id_rank", None)),
                               get(r, "adm2_pcode", None), calibrated)
        p = res["probs"]
        rows.append({"pred_category": pred.classes[p.argmax()], "pred_prob": round(float(p.max()), 4),
                     **{f"p_{c}": round(float(v), 4) for c, v in zip(pred.classes, p)}})
        print(f"  row {k + 1}/{len(src)}: {rows[-1]['pred_category']} ({rows[-1]['pred_prob']:.0%})")
    result = pd.concat([src, pd.DataFrame(rows)], axis=1)
    result.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {out} ({len(result)} rows)")


def interactive(pred, calibrated):
    print("\nWhat do you want to predict?")
    while True:
        print("\n  1  A new POI at a location (latitude, longitude and how it would be mapped)\n"
              "  2  An existing POI (node_id, OSM id such as node/123, or part of its name)\n"
              "  3  A CSV of new POIs (columns lat, lon; optional has_name, has_address, has_hours, footprint_m2, "
              "is_point, osm_id_rank, adm2_pcode)\n"
              f"  c  Switch calibrated class offsets {'off' if calibrated else 'on'} (now {'on' if calibrated else 'off'})\n"
              "  q  Quit")
        try:
            choice = ask("Choice", "1").lower()
            if choice == "q":
                return
            if choice == "c":
                calibrated = not calibrated
            elif choice == "1":
                lat = ask("  Latitude (e.g. -29.3151)", cast=float)
                lon = ask("  Longitude (e.g. 27.4869)", cast=float)
                has_name = yes("  Has a name?", "y")
                has_address = yes("  Has an address?")
                has_hours = yes("  Has opening hours?")
                fp = ask("  Building outline area in m2 (0 = mapped as a point only)", 0.0, float)
                show(pred.predict_new(lat, lon, has_name, has_address, has_hours, fp, calibrated=calibrated),
                     pred.classes, pred.class_f1)
            elif choice == "2":
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
                show(pred.predict_existing(hits[0], calibrated), pred.classes, pred.class_f1)
            elif choice == "3":
                path = ask("  Input CSV path")
                batch(pred, path, ask("  Output CSV path", str(Path(path).with_name(Path(path).stem + "_predictions.csv"))),
                      calibrated)
            else:
                print(f"  unknown choice {choice!r}")
        except (ValueError, OSError) as e:
            print(f"  error: {e}")
        except (EOFError, KeyboardInterrupt):
            print()
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calibrated", action="store_true", help="apply the validation-fitted class offsets "
                    "(model.ipynb Section 5; they did not improve the test score, so they are off by default)")
    ap.add_argument("--check", action="store_true", help="verify features and predictions against the notebook")
    sub = ap.add_subparsers(dest="mode")
    n = sub.add_parser("new", help="predict a new POI at a location")
    n.add_argument("--lat", type=float, required=True)
    n.add_argument("--lon", type=float, required=True)
    n.add_argument("--name", action="store_true", help="the POI has a name")
    n.add_argument("--address", action="store_true", help="the POI has an address")
    n.add_argument("--hours", action="store_true", help="the POI has opening hours")
    n.add_argument("--footprint-m2", type=float, default=0.0, help="building outline area; 0 = point only")
    n.add_argument("--also-point", action="store_true", help="mapped as a point as well as an outline")
    n.add_argument("--osm-id-rank", type=float, help="0-1 mapping-date rank; default: that of its surroundings")
    n.add_argument("--adm2-pcode", help="sub-district code, e.g. LSA12; default: that of the nearest POI")
    e = sub.add_parser("existing", help="predict an existing POI")
    e.add_argument("--query", required=True, help="node_id, OSM id (node/123, way/456) or part of a name")
    b = sub.add_parser("batch", help="predict every row of a CSV of new POIs")
    b.add_argument("csv")
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
    elif args.mode == "new":
        is_point = 1 if (args.also_point or args.footprint_m2 <= 0) else 0
        show(pred.predict_new(args.lat, args.lon, args.name, args.address, args.hours, args.footprint_m2, is_point,
                              args.osm_id_rank, args.adm2_pcode, args.calibrated), pred.classes, pred.class_f1)
    elif args.mode == "existing":
        hits = pred.find(args.query)
        if not hits:
            sys.exit(f"No POI matches {args.query!r}")
        if len(hits) > 1:
            print(f"{len(hits)} POIs match; showing the first. Use a node_id or OSM id to pick another.")
        show(pred.predict_existing(hits[0], args.calibrated), pred.classes, pred.class_f1)
    elif args.mode == "batch":
        batch(pred, args.csv, args.out or str(Path(args.csv).with_name(Path(args.csv).stem + "_predictions.csv")),
              args.calibrated)
    else:
        interactive(pred, args.calibrated)


if __name__ == "__main__":
    main()
