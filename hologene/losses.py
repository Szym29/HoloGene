"""Signature-stratified CORAL loss across slices."""
from __future__ import annotations

import numpy as np
import torch


def _mean_and_cov(z, eps=1e-5):
    mu = z.mean(dim=0)
    zc = z - mu
    denom = max(zc.size(0) - 1, 1)
    return mu, (zc.t() @ zc) / (denom + eps)


def _valid_groups(signature, slice_id, min_n):
    """Per signature, index lists of the slices with at least ``min_n`` spots."""
    sig = signature.detach().cpu().numpy().astype(np.int64, copy=False)
    sl = slice_id.detach().cpu().numpy()
    if np.issubdtype(sl.dtype, np.floating):
        # slice ids are normalised z values: encode unique (rounded) values
        _, sl = np.unique(np.round(sl.astype(np.float64), 6), return_inverse=True)
    sl = sl.astype(np.int64, copy=False)

    groups: dict[int, dict[int, list[int]]] = {}
    for i, (k, s) in enumerate(zip(sig, sl)):
        groups.setdefault(int(k), {}).setdefault(int(s), []).append(i)

    valid = []
    for per_slice in groups.values():
        if sum(len(v) for v in per_slice.values()) < 2 * min_n:
            continue
        kept = [v for v in per_slice.values() if len(v) >= min_n]
        if len(kept) >= 2:
            valid.append(kept)
    return valid


def signature_coral_loss(z, signature, slice_id, min_n=16):
    """Cross-slice alignment of latent statistics within each signature group. Returns (loss, n_terms)."""
    valid = _valid_groups(signature, slice_id, min_n)
    if not valid:
        return z.new_tensor(0.0), 0
    loss = z.new_tensor(0.0)
    n_terms = 0
    for slices in valid:
        stats = [_mean_and_cov(z[torch.as_tensor(idx, device=z.device, dtype=torch.long)]) for idx in slices]
        for i in range(len(stats)):
            for j in range(i + 1, len(stats)):
                mu_loss = (stats[i][0] - stats[j][0]).pow(2).mean()
                cov_loss = (stats[i][1] - stats[j][1]).pow(2).mean()
                loss = loss + mu_loss + cov_loss
                n_terms += 1
    return loss / n_terms, n_terms
