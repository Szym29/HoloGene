"""Reconstruction metrics: masked MSE, masked Cosine, Pearson, masked Spearman."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import rankdata

COSINE_EPS = 1e-5
MIN_POSITIVE_GENES = 2


def _dense_float64(x: Any) -> np.ndarray:
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float64)


def rowwise_correlation(a: np.ndarray, b: np.ndarray, masked: bool = False) -> np.ndarray:
    """Pearson correlation between matching rows of ``a`` and ``b``.

    With ``masked=True`` the inputs may contain NaN for excluded entries; they
    are centred with the NaN-mean and then contribute zero. Rows with zero
    variance get correlation 0. Results are clipped to [-1, 1].
    """
    if masked:
        a = a - np.nanmean(a, axis=1, keepdims=True)
        b = b - np.nanmean(b, axis=1, keepdims=True)
        a = np.nan_to_num(a)
        b = np.nan_to_num(b)
    else:
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    corr = np.divide((a * b).sum(1), den, out=np.zeros_like(den), where=den > 0)
    return np.clip(corr, -1, 1)


def reconstruction_metrics(truth: Any, prediction: Any, chunk_size: int = 256) -> dict[str, float | int]:
    """Compute masked MSE, masked Cosine, Pearson and masked Spearman.

    Parameters
    ----------
    truth, prediction
        Arrays (or sliceable array-likes, e.g. h5py datasets) of identical
        shape ``(n_spots, n_genes)``.
    chunk_size
        Number of spots processed at once.
    """
    if truth.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: truth {truth.shape} vs prediction {prediction.shape}")

    sse = count = 0.0
    masked_dot = masked_tt = masked_pp = 0.0
    dot = tt = pp = 0.0
    pearson = spearman = 0.0
    n_spots = 0
    for start in range(0, truth.shape[0], chunk_size):
        t = _dense_float64(truth[start:start + chunk_size])
        p = _dense_float64(prediction[start:start + chunk_size])
        if not (np.isfinite(t).all() and np.isfinite(p).all()):
            raise ValueError("Non-finite values in truth or prediction")
        keep = (t > 0).sum(1) >= MIN_POSITIVE_GENES
        t = t[keep]
        p = p[keep]
        n_spots += len(t)
        mask = t > 0

        t_pos, p_pos = t[mask], p[mask]
        sse += np.square(t_pos - p_pos).sum()
        count += mask.sum()
        masked_dot += (t_pos * p_pos).sum()
        masked_tt += (t_pos * t_pos).sum()
        masked_pp += (p_pos * p_pos).sum()

        dot += (t * p).sum()
        tt += (t * t).sum()
        pp += (p * p).sum()

        pearson += rowwise_correlation(t, p).sum()
        rank_t = rankdata(np.where(mask, t, np.nan), axis=1, nan_policy="omit")
        rank_p = rankdata(np.where(mask, p, np.nan), axis=1, nan_policy="omit")
        spearman += rowwise_correlation(rank_t, rank_p, masked=True).sum()

    if n_spots == 0:
        raise ValueError(f"No spot has >= {MIN_POSITIVE_GENES} ground-truth-positive genes")
    return {
        "masked_MSE": float(sse / count),
        "masked_Cosine": float(masked_dot / (np.sqrt(masked_tt * masked_pp) + COSINE_EPS)),
        "Pearson": float(pearson / n_spots),
        "masked_Spearman": float(spearman / n_spots),
        "overall_Cosine": float(dot / (np.sqrt(tt * pp) + COSINE_EPS)),
        "n_spots": int(n_spots),
        "n_genes": int(truth.shape[1]),
    }
