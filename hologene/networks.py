"""HoloGene networks: graph autoencoder (GAE) and the FiLM-conditioned INR."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ("full", "film_psn", "film_coral", "film", "xyz")
PSN_VARIANTS = ("full", "film_psn")
CORAL_VARIANTS = ("full", "film_coral")

# Seed for the extra z-row of the Fourier matrix in the isotropic XYZ variant.
XYZ_PROJECTION_SEED = 108848


# ---------------------------------------------------------------------------
# Graph autoencoder
# ---------------------------------------------------------------------------
class GraphConv(nn.Module):
    """Dense graph convolution ``adj @ (x @ W + b)``."""

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(in_features, out_features))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias
        return torch.mm(adj, x)


def _mlp(dim_in: int, dim_hidden: list[int], dim_out: int) -> nn.Sequential:
    """Linear-ReLU stack."""
    if len(dim_hidden) == 0:
        return nn.Sequential(nn.Linear(dim_in, dim_out))
    return nn.Sequential(
        nn.Linear(dim_in, dim_hidden[0]),
        nn.ReLU(),
        _mlp(dim_hidden[0], dim_hidden[1:], dim_out),
    )


class Encoder(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: list[int], dim_latent: int):
        super().__init__()
        self.net = _mlp(dim_in, dim_hidden, dim_latent)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Latent -> expression decoder with a Softplus(beta=5) output."""

    def __init__(self, dim_in: int, dim_hidden: list[int], dim_out: int):
        super().__init__()
        self.net = _mlp(dim_in, dim_hidden, dim_out)
        self.final_output_act_fn = nn.Softplus(beta=5)

    def forward(self, x):
        return self.final_output_act_fn(self.net(x))


class GAE(nn.Module):
    """Graph autoencoder (one dense GCN layer + MLP encoder, MLP decoder).

    ``dim_hidden = [h0, h1, ...]``: the GCN layer maps genes -> h0, the encoder
    maps h0 -> ... -> latent and the decoder mirrors the hidden sizes.
    """

    def __init__(self, dim_in: int, dim_hidden: list[int], dim_latent: int, in_logarithm: bool = False):
        super().__init__()
        self.gcn_layer = GraphConv(dim_in, dim_hidden[0])
        self.encoder = Encoder(dim_hidden[0], dim_hidden[1:], dim_latent)
        self.decoder = Decoder(dim_latent, dim_hidden[::-1], dim_in)
        self.in_logarithm = in_logarithm
        self.BN = nn.BatchNorm1d(dim_hidden[0])

    def forward(self, x, adj, idx):
        """Return (reconstruction, latent) for the target spots ``idx`` of a subgraph."""
        if self.in_logarithm:
            x = torch.log(1 + x)
        h = F.relu(self.gcn_layer(x, adj))
        h = h[idx, :]
        z = self.encoder(h)
        return self.decoder(z), z

    def forward_loss(self, x, adj, idx):
        y, z = self.forward(x, adj, idx)
        return F.mse_loss(y, x[idx, :]), y, z


def save_gae(path, gae: GAE, meta: dict) -> None:
    torch.save({"model_state_dict": gae.state_dict(), **meta}, path)


def load_gae(path, device="cpu") -> GAE:
    """Load a GAE checkpoint written by ``pretrain_gae.py`` (frozen, eval mode)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    gae = GAE(
        dim_in=ckpt["input_dim"],
        dim_hidden=ckpt.get("dim_hidden", [2048, 512]),
        dim_latent=ckpt.get("latent_dim", 64),
        in_logarithm=ckpt.get("in_logarithm", False),
    )
    gae.load_state_dict(ckpt["model_state_dict"])
    gae = gae.to(device).eval()
    for p in gae.parameters():
        p.requires_grad_(False)
    return gae


# ---------------------------------------------------------------------------
# HoloGene INR
# ---------------------------------------------------------------------------
class GaussianEncoding(nn.Module):
    """Random Fourier features: [sin(2 pi x B), cos(2 pi x B)], B ~ N(0, scale^2)."""

    def __init__(self, num_input_channels: int, mapping_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.register_buffer("B", torch.randn((num_input_channels, mapping_size)) * scale)

    def forward(self, x):
        x = 2 * np.pi * (x @ self.B.to(x.device))
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class HoloGeneINR(nn.Module):
    """Decoupled, FiLM-conditioned coordinate network producing latent embeddings."""

    def __init__(self, dim_in=2, dim_hidden=256, dim_out=64, num_layers=5,
                 fourier_scale=1.0, fourier_mapping_size=256,
                 n_signatures=30, signature_emb_dim=16, signature_film_weight=1.0,
                 variant="full"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
        self.variant = variant
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.n_celltypes = n_signatures
        self.ct_film_weight = float(signature_film_weight)

        self.fourier_transform = GaussianEncoding(dim_in, fourier_mapping_size, fourier_scale)
        fdim = dim_in * fourier_mapping_size
        self.linears = nn.ModuleList(
            [nn.Linear(fdim, dim_hidden)] + [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_layers - 1)]
        )
        self.last = nn.Linear(dim_hidden, dim_out)
        # Pioneer Signature Network: logits = ct_pred([gamma(x, y) || MLP_z(z)])
        self.ct_emb = nn.Embedding(n_signatures, signature_emb_dim)
        self.z_to_ct = nn.Sequential(nn.Linear(1, dim_hidden), nn.ReLU())
        self.ct_pred = nn.Sequential(
            nn.Linear(fdim + dim_hidden, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, n_signatures)
        )
        self.alpha_ct = nn.Parameter(torch.tensor(-1.0))
        self.film = nn.ModuleList([nn.Linear(1 + signature_emb_dim, dim_hidden * 2) for _ in range(num_layers)])
        self.film_z = nn.ModuleList([nn.Linear(1, dim_hidden * 2) for _ in range(num_layers)])
        self.film_ct = nn.ModuleList([nn.Linear(signature_emb_dim, dim_hidden * 2) for _ in range(num_layers)])

        self._configure_variant()

    # -- variant handling ---------------------------------------------------
    def _configure_variant(self):
        """Disable components for ablations."""
        if self.variant in PSN_VARIANTS:
            return
        self.ct_film_weight = 0.0
        frozen = ("ct_emb.", "z_to_ct.", "ct_pred.", "film_ct.", "film.")
        if self.variant == "xyz":
            frozen = frozen + ("film_z.",)
            # Isotropic input: append one Fourier row for z (same 2*256 features).
            b = self.fourier_transform.B.detach()
            rng = torch.Generator(device="cpu").manual_seed(XYZ_PROJECTION_SEED)
            z_row = torch.randn((1, b.shape[1]), generator=rng, dtype=b.dtype).to(b.device)
            self.fourier_transform.B = torch.cat([b, z_row], dim=0)
        for name, p in self.named_parameters():
            if name.startswith(frozen) or name == "alpha_ct":
                p.requires_grad_(False)

    @property
    def use_psn(self) -> bool:
        return self.variant in PSN_VARIANTS

    # -- PSN ----------------------------------------------------------------
    def _signature_logits(self, xy_features, z):
        return self.ct_pred(torch.cat([xy_features, self.z_to_ct(z)], dim=-1))

    def ct_logits(self, xyz):
        """PSN signature logits for coordinates ``xyz`` [B, 3] (zeros if PSN is off)."""
        if not self.use_psn:
            return xyz.new_zeros((len(xyz), self.n_celltypes))
        xy, z = xyz[:, :2], xyz[:, 2].reshape(-1, 1)
        return self._signature_logits(self.fourier_transform(xy), z)

    def predict_ct(self, xyz):
        """PSN signature probability distribution."""
        return torch.softmax(self.ct_logits(xyz), dim=-1)

    # -- synthesis ----------------------------------------------------------
    def forward(self, xy, z):
        if self.variant == "xyz":
            h = self.fourier_transform(torch.cat([xy, z], dim=-1))
            for linear in self.linears:
                h = torch.relu(linear(h))
            return self.last(h)

        h = self.fourier_transform(xy)
        if self.use_psn:
            probs = torch.softmax(self._signature_logits(h, z), dim=-1)
            signature = probs.detach() @ self.ct_emb.weight
            gate = torch.sigmoid(self.alpha_ct) * self.ct_film_weight
        for i, linear in enumerate(self.linears):
            h = linear(h)
            film = self.film_z[i](z)
            if self.use_psn:
                film = film + gate * self.film_ct[i](signature)
            scale, shift = film.chunk(2, dim=-1)
            h = torch.relu(scale * h + shift)
        return self.last(h)


class HoloGene(nn.Module):
    """Coordinate -> (latent, expression) model: INR followed by the GAE decoder."""

    def __init__(self, decoder: nn.Module, latent_dim=64, hidden_dim=256, num_layers=5,
                 n_signatures=30, signature_emb_dim=16, signature_film_weight=1.0,
                 fourier_scale=1.0, fourier_mapping_size=256, variant="full"):
        super().__init__()
        self.decoder = decoder
        self.inr = HoloGeneINR(
            dim_in=2, dim_hidden=hidden_dim, dim_out=latent_dim, num_layers=num_layers,
            fourier_scale=fourier_scale, fourier_mapping_size=fourier_mapping_size,
            n_signatures=n_signatures, signature_emb_dim=signature_emb_dim,
            signature_film_weight=signature_film_weight, variant=variant,
        )

    def forward(self, xyz):
        xy, z = xyz[:, :2], xyz[:, 2].reshape(xyz.shape[0], -1)
        latent = self.inr(xy, z)
        return latent, self.decoder(latent)

    def psn_head_parameters(self):
        """Parameters of the dedicated PSN classifier (trained with a smaller LR)."""
        return [p for n, p in self.named_parameters() if n.startswith(("inr.ct_pred.", "inr.z_to_ct."))]
