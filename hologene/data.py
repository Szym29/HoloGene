"""Data loading, preprocessing and slice-wise splitting.

The same preprocessing is shared by GAE pretraining and HoloGene training:

1. read the h5ad and drop spots with NaN coordinates;
2. optionally subsample spots (``sc.pp.subsample``, ``random_state=seed``);
3. keep the top-``n_top_genes`` highly variable genes (skipped when the file
   already contains exactly that many genes);
4. densify the expression matrix and normalise the 3D coordinates to [-1, 1]
   (aspect ratio preserved);
5. split the *slices* (unique z values) into train / held-out sets with
   ``train_test_split(test_size, random_state=split_seed)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from torch.utils.data import DataLoader, Dataset


class CoordNormalizer:
    """Map coordinates to [-1, 1] per axis, optionally preserving aspect ratio."""

    def __init__(self, coords, keep_ratio=True):
        coords = coords.astype("float32")
        self.keep_ratio = keep_ratio
        self.n_dims = coords.shape[1]
        self.min_vals = coords.min(axis=0)
        self.max_vals = coords.max(axis=0)
        self.ranges = self.max_vals - self.min_vals
        self.ranges[self.ranges == 0] = 1e-6
        self.scales = self.ranges / np.max(self.ranges)

    def normalize(self, coords):
        coords = coords.astype("float32").copy()
        for i in range(self.n_dims):
            coords[:, i] = (coords[:, i] - self.min_vals[i]) / self.ranges[i]
        coords -= 0.5
        coords *= 2.0
        if self.keep_ratio:
            for i in range(self.n_dims):
                coords[:, i] *= self.scales[i]
        return coords

    def denormalize(self, coords_norm):
        coords = coords_norm.astype("float32").copy()
        if self.keep_ratio:
            for i in range(self.n_dims):
                coords[:, i] /= self.scales[i]
        coords /= 2.0
        coords += 0.5
        for i in range(self.n_dims):
            coords[:, i] = coords[:, i] * self.ranges[i] + self.min_vals[i]
        return coords

    def state_dict(self):
        return dict(min_vals=self.min_vals, max_vals=self.max_vals, ranges=self.ranges,
                    scales=self.scales, keep_ratio=self.keep_ratio)


@dataclass
class DataConfig:
    h5ad: str
    signature_key: str | None = "leiden"
    subsample_fraction: float = 1.0
    subsample_seed: int | None = None   # defaults to the training seed
    n_top_genes: int = 3000
    keep_ratio: bool = True
    split: str = "random"               # "random" (test_fraction, split_seed) or "stride"
    test_fraction: float = 0.2
    split_seed: int = 8848
    stride: int = 2                     # split="stride": observe every stride-th section
    n_neighbors: int = 5
    perturbation: dict | None = None    # {translation_pct, rotation_deg, axial_jitter, seed}
    label_noise: dict | None = None     # {fraction, seed}: corrupt training pseudo-labels

    @classmethod
    def from_dict(cls, d: dict) -> "DataConfig":
        return cls(**d)


def perturb_sections(coords, translation_pct=0.0, rotation_deg=0.0, axial_jitter=0.0, seed=0):
    """Independent rigid perturbation of every section (registration stress test).

    For each section (in z order): rotation about its XY centroid with angle
    ~ N(0, rotation_deg), translation ~ N(0, translation_pct/100 * W / sqrt(2))
    per axis (W = largest XY extent), and a z shift ~ N(0, axial_jitter *
    median section spacing) clipped to stay between the neighbouring sections.
    """
    coords = np.asarray(coords, dtype=np.float64)
    out = coords.copy()
    rng = np.random.default_rng(seed)
    unique_z = np.unique(coords[:, 2])
    sections = np.searchsorted(unique_z, coords[:, 2])
    width = float(np.ptp(coords[:, :2], axis=0).max())
    section_z = np.asarray([np.median(coords[sections == s, 2]) for s in range(len(unique_z))])
    gaps = np.diff(np.sort(section_z))
    spacing = float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else 1.0
    margin = max(spacing * 1e-6, 1e-9)
    for s in range(len(unique_z)):
        mask = sections == s
        center = coords[mask, :2].mean(axis=0)
        theta = np.deg2rad(float(rng.normal(0.0, rotation_deg)))
        rot = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        shift = rng.normal(0.0, translation_pct / 100.0 * width / np.sqrt(2.0), size=2)
        out[mask, :2] = (coords[mask, :2] - center) @ rot.T + center + shift
        dz = float(rng.normal(0.0, axial_jitter * spacing))
        lower = (section_z[s - 1] + section_z[s]) / 2.0 if s > 0 else -np.inf
        upper = (section_z[s] + section_z[s + 1]) / 2.0 if s + 1 < len(section_z) else np.inf
        out[mask, 2] += float(np.clip(section_z[s] + dz, lower + margin, upper - margin)) - section_z[s]
    return out.astype(np.float32)


def corrupt_labels(labels, fraction, seed):
    """Reassign the labels of a random ``fraction`` of spots.

    The selected labels are permuted among themselves by a class-sorted cyclic
    shift, so every selected spot changes label and class counts are preserved.
    """
    labels = np.asarray(labels).copy()
    n = int(round(len(labels) * fraction))
    sel = np.sort(np.random.default_rng(seed).choice(len(labels), size=n, replace=False))
    vals = labels[sel]
    order = np.argsort(vals, kind="stable")
    shift = int(np.unique(vals, return_counts=True)[1].max())
    new = np.empty_like(vals)
    new[order] = np.roll(vals[order], shift)
    labels[sel] = new
    return labels


def load_and_preprocess(cfg: DataConfig, seed: int):
    """Steps 1-4 above. Returns (adata, normalizer)."""
    adata = sc.read_h5ad(cfg.h5ad)
    adata = adata[~np.isnan(adata.obsm["spatial"]).any(axis=1)]
    if cfg.subsample_fraction and cfg.subsample_fraction < 1.0:
        subsample_seed = seed if cfg.subsample_seed is None else cfg.subsample_seed
        adata = sc.pp.subsample(adata, fraction=cfg.subsample_fraction, copy=True, random_state=subsample_seed)
    if cfg.n_top_genes < adata.n_vars:
        sc.pp.highly_variable_genes(adata, n_top_genes=cfg.n_top_genes, subset=True)
    else:
        adata = adata.copy()
    adata.X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
    adata.X = adata.X.astype(np.float32, copy=False)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    normalizer = CoordNormalizer(coords, keep_ratio=cfg.keep_ratio)
    if cfg.perturbation:
        coords = perturb_sections(coords, **cfg.perturbation)
    adata.obsm["spatial"] = normalizer.normalize(coords.copy())
    return adata, normalizer


def add_signature_index(adata, key: str):
    """Integer-encode the pseudo-label column used for PSN supervision / CORAL.

    Labels are indexed in lexicographic order of their string values.
    """
    labels = adata.obs[key].astype(str)
    adata.obs["ct_index"] = np.argmax(labels.str.get_dummies().values, axis=1)
    return int(adata.obs["ct_index"].nunique())


def split_slices(adata, cfg: DataConfig):
    """Slice-wise split on unique z values. Returns (train, test, train_z, test_z).

    ``split="random"``: ``train_test_split(test_size=test_fraction, random_state=split_seed)``;
    ``split="stride"``: sections 0, k, 2k, ... (in z order) are observed, the rest held out.
    """
    unique_z = np.unique(adata.obsm["spatial"][:, 2])
    if cfg.split == "stride":
        train_z = unique_z[np.arange(0, len(unique_z), cfg.stride)]
        test_z = np.setdiff1d(unique_z, train_z)
    elif cfg.split == "random":
        train_z, test_z = train_test_split(unique_z, test_size=cfg.test_fraction, random_state=cfg.split_seed)
    else:
        raise ValueError(f"unknown split {cfg.split!r}")
    train = adata[np.isin(adata.obsm["spatial"][:, 2], train_z)].copy()
    test = adata[np.isin(adata.obsm["spatial"][:, 2], test_z)].copy()
    return train, test, train_z, test_z


# ---------------------------------------------------------------------------
# Spatial kNN graph utilities (GAE)
# ---------------------------------------------------------------------------
def spatial_knn_graph(coords, n_neighbors: int):
    """Binary kNN connectivity (self included) as float32 CSR."""
    return kneighbors_graph(coords, n_neighbors, mode="connectivity", n_jobs=8,
                            include_self=True).astype(np.float32)


class NeighborhoodDataset(Dataset):
    """Yields each spot index together with the indices of its kNN neighbourhood."""

    def __init__(self, adj, n_neighbors: int):
        self.neighbors = adj.indices.reshape(-1, n_neighbors)

    def __len__(self):
        return self.neighbors.shape[0]

    def __getitem__(self, idx):
        return {"idx": idx, "neighbors": self.neighbors[idx]}


def batch_subgraph(expression: torch.Tensor, adj, batch):
    """Induced subgraph on the union of the batch spots' neighbourhoods.

    Returns (subgraph expression, dense subgraph adjacency, positions of the
    batch spots inside the subgraph).
    """
    nodes = list(set(batch["neighbors"].cpu().numpy().flatten()))
    position = {node: i for i, node in enumerate(nodes)}
    target = [position[i] for i in batch["idx"].cpu().numpy()]
    sub_x = expression[nodes]
    sub_adj = torch.tensor(adj[nodes][:, nodes].toarray())
    return sub_x, sub_adj, target


@torch.no_grad()
def gae_embeddings(adata, gae, n_neighbors=5, batch_size=512, device="cuda"):
    """Embed every spot of ``adata`` with the frozen GAE encoder (kNN graph on ``adata``)."""
    adj = spatial_knn_graph(adata.obsm["spatial"], n_neighbors)
    expression = torch.tensor(adata.X, dtype=torch.float32)
    loader = DataLoader(NeighborhoodDataset(adj, n_neighbors), batch_size=batch_size, shuffle=False)
    gae = gae.to(device).eval()
    chunks = []
    for batch in loader:
        sub_x, sub_adj, target = batch_subgraph(expression, adj, batch)
        _, z = gae(sub_x.to(device), sub_adj.to(device), target)
        chunks.append(z.cpu())
    return torch.cat(chunks).numpy()


# ---------------------------------------------------------------------------
# HoloGene training data
# ---------------------------------------------------------------------------
class SpotDataset(Dataset):
    """(expression, xyz, GAE latent target, signature label, slice id) per spot."""

    def __init__(self, adata):
        self.expression = torch.tensor(adata.X, dtype=torch.float32)
        self.coords = torch.tensor(adata.obsm["spatial"], dtype=torch.float32)
        self.embedding = torch.tensor(adata.obsm["embedding"], dtype=torch.float32)
        self.signature = torch.tensor(adata.obs["ct_index"].to_numpy(), dtype=torch.long)
        self.slice = torch.tensor(adata.obsm["spatial"][:, 2], dtype=torch.float32)

    def __len__(self):
        return self.expression.shape[0]

    def __getitem__(self, idx):
        return (self.expression[idx], self.coords[idx], self.embedding[idx],
                self.signature[idx], self.slice[idx])
