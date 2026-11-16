"""HoloGene training (latent synthesis stage)."""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .data import (DataConfig, SpotDataset, add_signature_index, corrupt_labels, gae_embeddings,
                   load_and_preprocess, split_slices)
from .losses import signature_coral_loss
from .networks import CORAL_VARIANTS, HoloGene, load_gae


@dataclass
class ModelConfig:
    variant: str = "full"
    hidden_dim: int = 256
    num_layers: int = 5
    fourier_scale: float = 1.0
    fourier_mapping_size: int = 256
    signature_emb_dim: int = 16
    signature_film_weight: float = 0.02


@dataclass
class TrainConfig:
    seed: int = 8848
    epochs: int = 200
    warmup_epochs: int = 100
    batch_size: int = 8096
    lr: float = 5e-4
    psn_head_lr_multiplier: float = 0.2
    w_gex_start: float = 0.05
    w_gex_end: float = 2.0
    gex_ramp_epochs: int = 100
    w_psn: float = 0.001
    w_coral: float = 0.1
    coral_min_n: int = 16
    unfreeze_decoder_after_warmup: bool = True
    val_freq: int = 100
    gae_embed_batch_size: int = 512


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return dict(
        data=DataConfig.from_dict(cfg["data"]),
        gae_checkpoint=cfg["gae"]["checkpoint"],
        model=ModelConfig(**cfg.get("model", {})),
        train=TrainConfig(**cfg.get("training", {})),
        output_dir=cfg["output_dir"],
        raw=cfg,
    )


def gex_weight(epoch: int, cfg: TrainConfig) -> float:
    if epoch < cfg.warmup_epochs:
        return cfg.w_gex_start
    progress = min(1.0, (epoch - cfg.warmup_epochs) / max(1, cfg.gex_ramp_epochs))
    return cfg.w_gex_start + (cfg.w_gex_end - cfg.w_gex_start) * progress


def make_optimizer(model: HoloGene, lr: float, head_lr_multiplier: float):
    params = [p for p in model.parameters() if p.requires_grad]
    head_ids = {id(p) for p in model.psn_head_parameters()}
    heads = [p for p in params if id(p) in head_ids]
    if not heads or head_lr_multiplier == 1.0:
        return torch.optim.Adam(params, lr=lr)
    body = [p for p in params if id(p) not in head_ids]
    return torch.optim.Adam([{"params": body, "lr": lr}, {"params": heads, "lr": lr * head_lr_multiplier}], lr=lr)


@torch.no_grad()
def predict(model: HoloGene, coords, device, chunk=4096):
    """Return (expression, latent) for normalised coordinates ``coords`` [N, 3]."""
    model.eval()
    coords = np.asarray(coords, dtype=np.float32)
    expr, latent = [], []
    for start in range(0, len(coords), chunk):
        z, x = model(torch.as_tensor(coords[start:start + chunk], device=device))
        latent.append(z.cpu().numpy())
        expr.append(x.cpu().numpy())
    return np.concatenate(expr), np.concatenate(latent)


@torch.no_grad()
def heldout_summary(model: HoloGene, heldout, batch_size, device):
    """Masked MSE / Pearson on held-out slices (printed during training)."""
    was_training = model.training
    model.eval()
    gt, pred = [], []
    for x, loc, *_ in DataLoader(SpotDataset(heldout), batch_size=batch_size, shuffle=False):
        gt.append(x.numpy())
        pred.append(model(loc.to(device))[1].cpu().numpy())
    model.train(was_training)
    gt, pred = np.concatenate(gt), np.concatenate(pred)
    keep = (gt > 0).sum(1) >= 2
    gt, pred = gt[keep], pred[keep]
    mask = gt > 0
    gc = gt - gt.mean(1, keepdims=True)
    pc = pred - pred.mean(1, keepdims=True)
    pearson = (gc * pc).sum(1) / (np.sqrt((gc ** 2).sum(1) * (pc ** 2).sum(1)) + 1e-12)
    return dict(masked_MSE=float(((gt - pred)[mask] ** 2).mean()), pearson=float(np.mean(pearson)))


def fit(model: HoloGene, train, heldout, cfg: TrainConfig, device):
    variant = model.inr.variant
    w_psn = cfg.w_psn if model.inr.use_psn else 0.0
    w_coral = cfg.w_coral if variant in CORAL_VARIANTS else 0.0
    print(f"[HoloGene] variant={variant} w_psn={w_psn} w_coral={w_coral} "
          f"signature_film_weight={model.inr.ct_film_weight}")

    loader = DataLoader(SpotDataset(train), batch_size=cfg.batch_size, shuffle=True)
    model = model.to(device)
    for p in model.decoder.parameters():
        p.requires_grad_(False)
    optimizer = make_optimizer(model, cfg.lr, cfg.psn_head_lr_multiplier)
    mse = nn.MSELoss()
    history = []

    for epoch in range(cfg.epochs):
        if cfg.unfreeze_decoder_after_warmup and epoch == cfg.warmup_epochs:
            for p in model.decoder.parameters():
                p.requires_grad_(True)
            optimizer = make_optimizer(model, cfg.lr, cfg.psn_head_lr_multiplier)
        w_gex = gex_weight(epoch, cfg)
        totals = dict(loss=0.0, emb=0.0, gex=0.0, psn=0.0, coral=0.0)
        n_batches = 0
        for expression, loc, target, signature, slice_id in loader:
            expression, loc, target = expression.to(device), loc.to(device), target.to(device)
            signature, slice_id = signature.to(device), slice_id.to(device)
            optimizer.zero_grad()

            latent, pred = model(loc)
            loss_gex = mse(pred, expression)
            loss_emb = mse(latent, target)
            loss = loss_emb + w_gex * loss_gex
            if w_psn > 0:
                loss_psn = F.cross_entropy(model.inr.ct_logits(loc), signature)
                loss = loss + w_psn * loss_psn
                totals["psn"] += loss_psn.item()
            if w_coral > 0:
                loss_coral, _ = signature_coral_loss(latent, signature, slice_id, min_n=cfg.coral_min_n)
                loss = loss + w_coral * loss_coral
                totals["coral"] += loss_coral.item()
            loss.backward()
            optimizer.step()

            totals["loss"] += loss.item()
            totals["emb"] += loss_emb.item()
            totals["gex"] += loss_gex.item()
            n_batches += 1

        history.append({k: v / n_batches for k, v in totals.items()})
        if (epoch + 1) % cfg.val_freq == 0:
            msg = " ".join(f"{k}={v:.5f}" for k, v in history[-1].items())
            print(f"[epoch {epoch + 1}/{cfg.epochs}] {msg}  w_gex={w_gex:.3f}", flush=True)
            if heldout is not None:
                print(f"    held-out (logging only): {heldout_summary(model, heldout, cfg.batch_size, device)}",
                      flush=True)
    return model, history


def run(config_path: str, device: str | None = None):
    """Full HoloGene pipeline: preprocess -> GAE targets -> train -> predict held-out slices."""
    cfg = load_config(config_path)
    tcfg, mcfg, dcfg = cfg["train"], cfg["model"], cfg["data"]
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    gae = load_gae(cfg["gae_checkpoint"], device=device)

    adata, normalizer = load_and_preprocess(dcfg, seed=tcfg.seed)
    n_signatures = add_signature_index(adata, dcfg.signature_key)
    train, heldout, train_z, test_z = split_slices(adata, dcfg)
    if dcfg.label_noise:
        train.obs["ct_index"] = corrupt_labels(train.obs["ct_index"].to_numpy(), **dcfg.label_noise)
    print(f"[data] {adata.n_obs} spots x {adata.n_vars} genes, {len(train_z)} train / {len(test_z)} "
          f"held-out slices, {train.n_obs} / {heldout.n_obs} spots, {n_signatures} signatures")

    train.obsm["embedding"] = gae_embeddings(train, gae, dcfg.n_neighbors, tcfg.gae_embed_batch_size, device)
    heldout.obsm["embedding"] = gae_embeddings(heldout, gae, dcfg.n_neighbors, tcfg.gae_embed_batch_size, device)

    model = HoloGene(
        decoder=copy.deepcopy(gae.decoder), latent_dim=heldout.obsm["embedding"].shape[1],
        hidden_dim=mcfg.hidden_dim, num_layers=mcfg.num_layers, n_signatures=n_signatures,
        signature_emb_dim=mcfg.signature_emb_dim, signature_film_weight=mcfg.signature_film_weight,
        fourier_scale=mcfg.fourier_scale, fourier_mapping_size=mcfg.fourier_mapping_size,
        variant=mcfg.variant,
    )
    model, history = fit(model, train, heldout, tcfg, device)

    expression, latent = predict(model, heldout.obsm["spatial"], device)
    result = heldout.copy()
    result.obsm["prediction"] = expression
    result.obsm["inr_embeddings"] = latent
    result.write_h5ad(os.path.join(out, "prediction.h5ad"))

    gae_ckpt = torch.load(cfg["gae_checkpoint"], map_location="cpu", weights_only=False)
    torch.save(dict(
        model_state_dict=model.state_dict(),
        model_config=asdict(mcfg), n_signatures=n_signatures,
        signature_categories=sorted(adata.obs[dcfg.signature_key].astype(str).unique().tolist()),
        gae=dict(input_dim=gae_ckpt["input_dim"], dim_hidden=gae_ckpt["dim_hidden"],
                 latent_dim=gae_ckpt["latent_dim"]),
        coord_normalizer=normalizer.state_dict(), genes=adata.var_names.tolist(),
    ), os.path.join(out, "hologene.pt"))
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg["raw"], f, sort_keys=False)
    z_all = np.unique(adata.obsm["spatial"][:, 2])
    with open(os.path.join(out, "split.json"), "w") as f:
        json.dump(dict(train_z=sorted(map(float, train_z)), test_z=sorted(map(float, test_z)),
                       train_slice_indices=np.flatnonzero(np.isin(z_all, train_z)).tolist(),
                       test_slice_indices=np.flatnonzero(np.isin(z_all, test_z)).tolist(),
                       n_train_spots=int(train.n_obs), n_test_spots=int(heldout.n_obs)), f, indent=2)
    with open(os.path.join(out, "training_history.json"), "w") as f:
        json.dump(history, f)
    print(f"[done] outputs written to {out}")
    return model, result
