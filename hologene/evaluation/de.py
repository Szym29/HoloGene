"""Annotation-wise one-vs-rest differential expression."""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

EXCLUDED_LABELS = ("nan", "None", "NA", "")
LOGFC_EPS = 1e-6


def valid_labels(labels: np.ndarray, min_label_cells: int = 100,
                 exclude: Sequence[str] = EXCLUDED_LABELS) -> list[str]:
    """Sorted labels that have at least ``min_label_cells`` spots."""
    counts = pd.Series(labels, dtype="string").value_counts()
    keep = sorted(str(label) for label, n in counts.items()
                  if int(n) >= min_label_cells and str(label) not in set(exclude))
    if len(keep) < 2:
        raise ValueError(f"Need at least two labels with >= {min_label_cells} spots; found {len(keep)}")
    return keep


def make_unique(names: Sequence[str]) -> list[str]:
    """Append ``__dup<k>`` to repeated names so they can serve as var_names."""
    seen: dict[str, int] = {}
    out = []
    for name in map(str, names):
        k = seen.get(name, 0)
        seen[name] = k + 1
        out.append(name if k == 0 else f"{name}__dup{k}")
    return out


def _scanpy_rank_genes(matrix: np.ndarray, labels: np.ndarray, groups: list[str],
                       gene_ids: list[str], method: str) -> pd.DataFrame:
    tmp = ad.AnnData(X=np.asarray(matrix, dtype=np.float32))
    tmp.var_names = pd.Index(np.asarray(gene_ids, dtype=str))
    tmp.obs["label"] = pd.Categorical(labels, categories=groups)
    tmp.obs_names_make_unique()
    with warnings.catch_warnings():  # scanpy builds its result frame column by column
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        sc.tl.rank_genes_groups(tmp, groupby="label", method=method, n_genes=tmp.n_vars, use_raw=False)
    frames = []
    for group in groups:
        frame = sc.get.rank_genes_groups_df(tmp, group=group)
        frame["label"] = group
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True).rename(columns={
        "names": "gene_id", "scores": "score", "logfoldchanges": "scanpy_logfoldchange",
        "pvals": "pval", "pvals_adj": "pval_adj"})
    return out[["label", "gene_id", "score", "scanpy_logfoldchange", "pval", "pval_adj"]]


def one_vs_rest_de(matrix: Any, labels: Sequence[str], gene_symbols: Sequence[str], *,
                   min_label_cells: int = 100, method: str = "t-test_overestim_var",
                   clip_negative: bool = True) -> pd.DataFrame:
    """One-vs-rest DE table for every retained annotation label.

    Parameters
    ----------
    matrix
        Expression matrix ``(n_spots, n_genes)`` (ground truth or prediction).
    labels
        Annotation label of every spot.
    gene_symbols
        Gene symbol of every column.

    Returns
    -------
    DataFrame with one row per (label, gene): ``label, gene_id, gene_symbol,
    logfc_manual, mean_in_label, mean_out_label, n_cells_label, n_cells_rest,
    score, scanpy_logfoldchange, pval, pval_adj``.
    """
    labels = np.asarray(labels).astype(str)
    gene_symbols = np.asarray(gene_symbols).astype(str)
    groups = valid_labels(labels, min_label_cells)
    rows = np.sort(np.concatenate([np.flatnonzero(labels == g) for g in groups]))
    x = np.asarray(matrix[rows, :])
    x = x.astype(np.float64, copy=False)
    if clip_negative:
        x = np.maximum(x, 0.0)
    labels = labels[rows]

    gene_ids = make_unique(gene_symbols)
    ranked = _scanpy_rank_genes(x, labels, groups, gene_ids, method)

    blocks = []
    n_genes = len(gene_ids)
    for group in groups:
        inside = labels == group
        mean_in = np.asarray(x[inside].mean(axis=0)).ravel().astype(np.float64)
        mean_out = np.asarray(x[~inside].mean(axis=0)).ravel().astype(np.float64)
        blocks.append(pd.DataFrame({
            "label": [group] * n_genes,
            "gene_id": gene_ids,
            "gene_symbol": gene_symbols.tolist(),
            "logfc_manual": np.log1p(mean_in + LOGFC_EPS) - np.log1p(mean_out + LOGFC_EPS),
            "mean_in_label": mean_in,
            "mean_out_label": mean_out,
            "n_cells_label": int(inside.sum()),
            "n_cells_rest": int((~inside).sum()),
        }))
    table = pd.concat(blocks, ignore_index=True)
    return table.merge(ranked, on=["label", "gene_id"], how="left")


def select_up_degs(de: pd.DataFrame, q_threshold: float = 0.05, logfc_threshold: float = 0.025,
                   top_n: int = 50) -> pd.DataFrame:
    """Significant up-regulated genes per label used as g:Profiler queries."""
    picked = []
    for label, sub in de.groupby("label", sort=True):
        hit = sub[(sub["pval_adj"].fillna(1.0) < q_threshold) & (sub["logfc_manual"] > logfc_threshold)]
        hit = hit.sort_values(["pval_adj", "logfc_manual", "score"], ascending=[True, False, False]).head(top_n)
        if not hit.empty:
            picked.append(hit.assign(direction="up"))
    if not picked:
        return pd.DataFrame(columns=list(de.columns) + ["direction"])
    return pd.concat(picked, ignore_index=True)
