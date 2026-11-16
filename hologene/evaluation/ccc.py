"""Cell-cell communication (CCC) Pearson and Top-F1 with CellTalkDB mouse ligand-receptor pairs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import kneighbors_graph

from .de import valid_labels

DEFAULT_LR_PAIRS = Path(__file__).resolve().parent / "resources" / "celltalkdb_mouse_lr_pairs.tsv"
NEIGHBORS = 15
TOP_FRACTION = 0.05
MIN_TOP = 25
MAX_TOP = 5000


def load_lr_pairs(path: str | Path = DEFAULT_LR_PAIRS) -> pd.DataFrame:
    """Ligand-receptor table with columns ``name, ligand, receptor, evidence``."""
    return pd.read_csv(path, sep="\t")


def match_lr_pairs(lr_pairs: pd.DataFrame, gene_symbols: Sequence[str]) -> tuple[list[tuple[int, int]], list[int]]:
    """Pairs present in the panel.

    Returns ``(pairs, columns)``: ``columns`` are the panel indices of all genes
    used (in order of first appearance) and every pair is given as
    ``(ligand_position, receptor_position)`` into ``columns``.
    """
    lookup: dict[str, int] = {}
    for idx, gene in enumerate(gene_symbols):
        key = str(gene).strip().upper()
        if key and key not in lookup:
            lookup[key] = idx
    columns: list[int] = []
    position: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    for ligand, receptor in zip(lr_pairs["ligand"].astype(str), lr_pairs["receptor"].astype(str)):
        lig, rec = ligand.strip().upper(), receptor.strip().upper()
        if lig not in lookup or rec not in lookup:
            continue
        for idx in (lookup[lig], lookup[rec]):
            if idx not in position:
                position[idx] = len(columns)
                columns.append(idx)
        pairs.append((position[lookup[lig]], position[lookup[rec]]))
    return pairs, columns


def label_means(matrix: Any, labels: np.ndarray, groups: list[str], columns: list[int],
                chunk_size: int = 2048) -> np.ndarray:
    """Mean (negatives clipped to 0) of the selected genes per label, ``(n_labels, n_columns)``."""
    code_of = {g: i for i, g in enumerate(groups)}
    rows = np.flatnonzero(np.asarray([label in code_of for label in labels], dtype=bool))
    codes = np.asarray([code_of[str(labels[i])] for i in rows], dtype=np.int64)
    cols = np.asarray(columns, dtype=np.int64)
    sums = np.zeros((len(groups), len(cols)), dtype=np.float64)
    counts = np.bincount(codes, minlength=len(groups)).astype(np.float64)
    for start in range(0, rows.size, chunk_size):
        values = np.asarray(matrix[rows[start:start + chunk_size], :])[:, cols].astype(np.float64, copy=False)
        np.add.at(sums, codes[start:start + chunk_size], np.maximum(values, 0.0))
    counts[counts == 0] = np.nan
    return sums / counts[:, None]


def label_adjacency(coords: np.ndarray, labels: np.ndarray, groups: list[str],
                    neighbors: int = NEIGHBORS) -> np.ndarray:
    """Label-to-label spatial adjacency ``A[s, r]`` from a row-normalised kNN graph."""
    code_of = {g: i for i, g in enumerate(groups)}
    keep = np.asarray([label in code_of for label in labels], dtype=bool)
    xyz = np.asarray(coords[keep], dtype=np.float64)
    codes = np.asarray([code_of[str(label)] for label in labels[keep]], dtype=np.int64)
    graph = kneighbors_graph(xyz, n_neighbors=neighbors, mode="connectivity", include_self=False,
                             n_jobs=8).astype(np.float64).tocsr()
    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sums)
    inv[row_sums > 0] = 1.0 / row_sums[row_sums > 0]
    graph = sparse.csr_matrix(graph.multiply(inv[:, None]))
    indicator = sparse.csr_matrix((np.ones(codes.size), (np.arange(codes.size), codes)),
                                  shape=(codes.size, len(groups)))
    counts = np.bincount(codes, minlength=len(groups)).astype(np.float64)
    counts[counts == 0] = np.nan
    return np.asarray((indicator.T @ graph @ indicator).toarray(), dtype=np.float64) / counts[:, None]


def communication_scores(means: np.ndarray, adjacency: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Flattened ``S[p, s, r]`` for all pairs (pair-major, then sender, receiver)."""
    return np.concatenate([(means[:, lig][:, None] * means[:, rec][None, :] * adjacency).ravel()
                           for lig, rec in pairs])


def ccc_metrics(truth: Any, prediction: Any, labels: Sequence[str], coords: np.ndarray,
                gene_symbols: Sequence[str], *, lr_pairs: pd.DataFrame | None = None,
                min_label_cells: int = 100, neighbors: int = NEIGHBORS) -> dict[str, Any]:
    """CCC Pearson and CCC Top-F1 between ground truth and prediction."""
    labels = np.asarray(labels).astype(str)
    groups = valid_labels(labels, min_label_cells)
    pairs, columns = match_lr_pairs(load_lr_pairs() if lr_pairs is None else lr_pairs, gene_symbols)
    if not pairs:
        raise ValueError("No ligand-receptor pair is fully contained in the gene panel")
    adjacency = label_adjacency(coords, labels, groups, neighbors)
    pred = communication_scores(label_means(prediction, labels, groups, columns), adjacency, pairs)
    true = communication_scores(label_means(truth, labels, groups, columns), adjacency, pairs)
    union = (pred > 0) | (true > 0)
    pred, true = pred[union], true[union]
    k = min(len(true), MAX_TOP, max(MIN_TOP, math.ceil(TOP_FRACTION * len(true))))
    pearson = (float(np.corrcoef(pred, true)[0, 1])
               if len(true) > 1 and np.std(true) > 0 and np.std(pred) > 0 else None)
    overlap = len(set(np.argsort(-pred, kind="stable")[:k]) & set(np.argsort(-true, kind="stable")[:k]))
    return {
        "CCC_Pearson": pearson,
        "CCC_Top_F1": overlap / k if k else None,
        "CCC_topK": int(k),
        "CCC_overlap": int(overlap),
        "CCC_LR_pairs": len(pairs),
        "CCC_candidate_union": int(len(true)),
    }
