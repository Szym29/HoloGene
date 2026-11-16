"""Top-20 DEG micro F1."""

from __future__ import annotations

import pandas as pd

TOP_N = 20


def top_up_degs(de: pd.DataFrame, top_n: int = TOP_N) -> dict[str, set[str]]:
    """Top ``top_n`` up-regulated gene symbols per label."""
    out: dict[str, set[str]] = {}
    for label, sub in de.groupby(de["label"].astype(str), sort=True):
        picked = sub[sub["logfc_manual"] > 0]
        picked = picked.sort_values(["logfc_manual", "score"], ascending=[False, False]).head(top_n)
        out[str(label)] = set(picked["gene_symbol"].astype(str))
    return out


def micro_f1(truth_sets: dict[str, set], pred_sets: dict[str, set]) -> dict[str, float | int]:
    """Micro-averaged precision/recall/F1 of per-label set overlaps."""
    overlap = n_truth = n_pred = 0
    for label in sorted(set(truth_sets) | set(pred_sets)):
        gt = truth_sets.get(label, set())
        pr = pred_sets.get(label, set())
        overlap += len(gt & pr)
        n_truth += len(gt)
        n_pred += len(pr)
    recall = overlap / n_truth if n_truth else float("nan")
    precision = overlap / n_pred if n_pred else float("nan")
    if n_truth and n_pred and recall + precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    elif n_truth or n_pred:
        f1 = 0.0
    else:
        f1 = float("nan")
    return {"f1": float(f1), "precision": float(precision), "recall": float(recall),
            "overlap": int(overlap), "n_truth": int(n_truth), "n_pred": int(n_pred)}


def deg_f1(truth_de: pd.DataFrame, pred_de: pd.DataFrame, top_n: int = TOP_N) -> dict[str, float | int]:
    """DEG micro F1 between ground-truth and predicted DE tables."""
    m = micro_f1(top_up_degs(truth_de, top_n), top_up_degs(pred_de, top_n))
    return {"DEG_F1": m["f1"], "DEG_precision": m["precision"], "DEG_recall": m["recall"],
            "DEG_overlap": m["overlap"], "DEG_GT_genes": m["n_truth"], "DEG_pred_genes": m["n_pred"]}
