"""Graph-autoencoder pretraining."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import (DataConfig, NeighborhoodDataset, batch_subgraph, load_and_preprocess,
                   spatial_knn_graph, split_slices)
from .networks import GAE, save_gae


@dataclass
class GAEConfig:
    hidden_dims: list = field(default_factory=lambda: [2048, 512])
    latent_dim: int = 64
    in_logarithm: bool = False


@dataclass
class GAETrainConfig:
    seed: int = 8848
    epochs: int = 200
    batch_size: int = 512
    lr: float = 1e-5
    log_freq: int = 20


def train_gae(train, gae_cfg: GAEConfig, cfg: GAETrainConfig, n_neighbors: int, device):
    adj = spatial_knn_graph(train.obsm["spatial"], n_neighbors)
    expression = torch.tensor(train.X)
    gae = GAE(train.n_vars, gae_cfg.hidden_dims, gae_cfg.latent_dim, in_logarithm=gae_cfg.in_logarithm).to(device)
    optimizer = torch.optim.Adam(gae.parameters(), lr=cfg.lr)
    loader = DataLoader(NeighborhoodDataset(adj, n_neighbors), batch_size=cfg.batch_size, shuffle=True)
    losses = []
    for epoch in range(cfg.epochs):
        gae.train()
        total, n = 0.0, 0
        for batch in loader:
            sub_x, sub_adj, target = batch_subgraph(expression, adj, batch)
            optimizer.zero_grad()
            loss, _, _ = gae.forward_loss(sub_x.to(device), sub_adj.to(device), target)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        losses.append(total / n)
        if (epoch + 1) % cfg.log_freq == 0:
            print(f"[GAE epoch {epoch + 1}/{cfg.epochs}] loss={losses[-1]:.6f}", flush=True)
    return gae, losses


def run(config_path: str, device: str | None = None):
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    dcfg = DataConfig.from_dict(raw["data"])
    gae_cfg = GAEConfig(**raw.get("gae", {}))
    tcfg = GAETrainConfig(**raw.get("training", {}))
    out = raw["output_dir"]
    os.makedirs(out, exist_ok=True)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    adata, _ = load_and_preprocess(dcfg, seed=tcfg.seed)
    train, heldout, train_z, test_z = split_slices(adata, dcfg)
    print(f"[data] {adata.n_obs} spots x {adata.n_vars} genes; GAE trained on {len(train_z)} slices "
          f"({train.n_obs} spots), {len(test_z)} slices held out")

    gae, losses = train_gae(train, gae_cfg, tcfg, dcfg.n_neighbors, device)
    save_gae(os.path.join(out, "gae.pt"), gae, dict(
        input_dim=adata.n_vars, latent_dim=gae_cfg.latent_dim, dim_hidden=gae_cfg.hidden_dims,
        in_logarithm=gae_cfg.in_logarithm, n_neighbors=dcfg.n_neighbors, final_loss=losses[-1], configs=raw))
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
    with open(os.path.join(out, "split.json"), "w") as f:
        json.dump(dict(train_z=sorted(map(float, train_z)), test_z=sorted(map(float, test_z))), f, indent=2)
    with open(os.path.join(out, "gene_names.txt"), "w") as f:
        f.write("\n".join(adata.var_names) + "\n")
    np.save(os.path.join(out, "training_losses.npy"), np.asarray(losses))
    print(f"[done] GAE written to {os.path.join(out, 'gae.pt')}")
    return gae
