"""Load a trained HoloGene model and query it at arbitrary 3D coordinates.

Example -- a virtual slice halfway between two measured slices::

    from hologene.inference import load_hologene, query
    model, info = load_hologene("outputs/mosta/split_8to2/hologene_full/hologene.pt")
    xy = adata_slice.obsm["spatial"][:, :2]            # raw (registered) x, y of a slice
    xyz = np.c_[xy, np.full(len(xy), z_between)]        # raw z between two sections
    expression, latent = query(model, info, xyz)        # genes in info["genes"] order
"""
from __future__ import annotations

import numpy as np
import torch

from .networks import Decoder, HoloGene
from .trainer import predict


def load_hologene(path, device=None):
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    g = ckpt["gae"]
    decoder = Decoder(g["latent_dim"], list(g["dim_hidden"])[::-1], g["input_dim"])
    m = ckpt["model_config"]
    model = HoloGene(decoder, latent_dim=g["latent_dim"], hidden_dim=m["hidden_dim"], num_layers=m["num_layers"],
                     n_signatures=ckpt["n_signatures"], signature_emb_dim=m["signature_emb_dim"],
                     signature_film_weight=m["signature_film_weight"], fourier_scale=m["fourier_scale"],
                     fourier_mapping_size=m["fourier_mapping_size"], variant=m["variant"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    info = dict(device=device, genes=ckpt["genes"], coord_normalizer=ckpt["coord_normalizer"],
                signature_categories=ckpt["signature_categories"])
    return model, info


def normalize_coords(xyz_raw, normalizer: dict):
    """Apply the training-time [-1, 1] coordinate normalisation to raw coordinates."""
    xyz = np.asarray(xyz_raw, dtype=np.float32).copy()
    xyz = (xyz - normalizer["min_vals"]) / normalizer["ranges"]
    xyz = (xyz - 0.5) * 2.0
    if normalizer["keep_ratio"]:
        xyz = xyz * normalizer["scales"]
    return xyz.astype(np.float32)


def query(model, info, xyz_raw, normalized=False):
    """Predict (expression, latent) at coordinates given in raw units (or normalised ones)."""
    xyz = np.asarray(xyz_raw, dtype=np.float32) if normalized else normalize_coords(xyz_raw, info["coord_normalizer"])
    return predict(model, xyz, info["device"])
