"""Hypothesis-level text verifier (design C, spec
docs/superpowers/specs/2026-11-07-language-verified-hypotheses-design.md): region geometry
of the egocentric descriptor, OSM region-fraction descriptors, greedy NMS of the image
posterior, descriptor similarity, listwise target, basin re-weighting of the posterior and
the caption -> descriptor verifier (frozen T5 tokens -> region queries)."""
import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .egocentric import coarse_grid_xz, fov_mask
from .voting import log_softmax_spatial


# ---------------------------------------------------------------------------
# frozen caption encoder
# ---------------------------------------------------------------------------
class TextEncoder(nn.Module):
    """Frozen pre-trained text encoder (default t5-base) returning token embeddings."""

    def __init__(self, name: str = "t5-base", max_tokens: int = 64):
        super().__init__()
        from transformers import AutoTokenizer, T5EncoderModel

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.encoder = T5EncoderModel.from_pretrained(name).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.max_tokens = max_tokens
        self.dim = self.encoder.config.d_model

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # always frozen
        return self

    @torch.no_grad()
    def forward(self, captions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.tokenizer(
            list(captions), return_tensors="pt", padding=True, truncation=True, max_length=self.max_tokens
        )
        device = next(self.encoder.parameters()).device
        ids, mask = tok["input_ids"].to(device), tok["attention_mask"].to(device)
        with torch.autocast("cuda", enabled=False):
            h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return h.float(), mask.bool()


# ---------------------------------------------------------------------------
# geometry and descriptors
# ---------------------------------------------------------------------------
def region_geometry(
    sectors: int, ranges: int, z_max: float, x_max: float, cell: float, fov_deg: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coarse camera-frame grid [h, w, 2], region id of every cell [h, w] (sector * ranges
    + range bin, -1 outside the FOV wedge) and region centres [R, 2] = (mean bearing in
    rad, mean range / z_max) of the cells of every region."""
    grid = coarse_grid_xz(z_max, x_max, cell)
    valid = fov_mask(grid, fov_deg, z_max)
    x, z = grid[..., 0], grid[..., 1]
    half = math.radians(fov_deg / 2)
    phi = torch.atan2(x, z)
    sct = ((phi + half) / (2 * half) * sectors).floor().clamp(0, sectors - 1)
    rr = torch.sqrt(x**2 + z**2) / z_max
    rng = (rr * ranges).floor().clamp(0, ranges - 1)
    region = torch.where(valid, (sct * ranges + rng).long(), torch.full(grid.shape[:2], -1, dtype=torch.long))
    R = sectors * ranges
    centres = torch.zeros(R, 2)
    for r in range(R):
        cells = region == r
        assert cells.any(), f"region {r} has no cell: too many sectors / ranges for this grid"
        centres[r, 0], centres[r, 1] = phi[cells].mean(), rr[cells].mean()
    return grid, region, centres


def region_fractions(onehot: torch.Tensor, region: torch.Tensor, R: int) -> torch.Tensor:
    """Class fractions per region: onehot [N, C, h, w], region [h, w] -> [N, R, C]."""
    reg = F.one_hot(region.clamp(min=0), R).to(onehot) * (region >= 0)[..., None].to(onehot)  # [h, w, R]
    counts = reg.sum((0, 1)).clamp(min=1)  # [R]
    return torch.einsum("nchw,hwr->nrc", onehot, reg) / counts[None, :, None]


def mirror_regions(x: torch.Tensor, sectors: int, ranges: int) -> torch.Tensor:
    """Left-right mirror of a region descriptor [..., R, C]: sector s <-> sectors - 1 - s."""
    perm = torch.tensor(
        [(sectors - 1 - s) * ranges + m for s in range(sectors) for m in range(ranges)], device=x.device
    )
    return x.index_select(-2, perm)


# ---------------------------------------------------------------------------
# modes of the image posterior
# ---------------------------------------------------------------------------
def nms_modes(log_probs: torch.Tensor, K: int, r_px: int, r_bins: int):
    """Greedy non-maximum suppression of a log-prob volume [B, H, W, N] (-inf = masked).
    Returns modes [B, K, 3] long (x, y, rotation index), their log-probs [B, K] (-inf when
    padded), valid [B, K] and basin_id [B, H, W, N] long (-1 = no basin; basins are
    disjoint, first mode wins, rotation window circular)."""
    B, H, W, N = log_probs.shape
    dev = log_probs.device
    modes = torch.zeros(B, K, 3, dtype=torch.long, device=dev)
    logp = torch.full((B, K), -np.inf, dtype=log_probs.dtype, device=dev)
    valid = torch.zeros(B, K, dtype=torch.bool, device=dev)
    basin = torch.full((B, H, W, N), -1, dtype=torch.long, device=dev)
    ii = torch.arange(H, device=dev)[:, None]
    jj = torch.arange(W, device=dev)[None, :]
    rr = torch.arange(N, device=dev)
    for b in range(B):
        lp = log_probs[b].clone()
        for k in range(K):
            flat = lp.flatten().argmax()
            v = lp.flatten()[flat]
            if torch.isinf(v):
                break
            y = torch.div(flat, W * N, rounding_mode="floor")
            x = torch.div(flat % (W * N), N, rounding_mode="floor")
            r = flat % N
            modes[b, k] = torch.stack([x, y, r])
            logp[b, k], valid[b, k] = v, True
            d2 = (ii - y) ** 2 + (jj - x) ** 2
            dr = (rr - r).abs() % N
            dr = torch.minimum(dr, N - dr)
            window = (d2 <= r_px**2)[:, :, None] & (dr <= r_bins)[None, None, :]
            basin[b][window & (basin[b] < 0)] = k
            lp[window] = -np.inf
    return modes, logp, valid, basin


def descriptor_similarity(d_hat: torch.Tensor, d_modes: torch.Tensor, region_w: torch.Tensor) -> torch.Tensor:
    """sum_r softmax(w)_r (1 - 1/2 ||d_hat_r - d_{k,r}||_1): d_hat [B, R, C], d_modes [B, K, R, C] -> [B, K]."""
    w = torch.softmax(region_w, 0)
    l1 = (d_hat[:, None] - d_modes).abs().sum(-1)  # [B, K, R]
    return ((1 - 0.5 * l1) * w).sum(-1)


def listwise_target(
    modes: torch.Tensor, valid: torch.Tensor, xy_gt: torch.Tensor, yaw_gt: torch.Tensor,
    num_rotations: int, ppm: float, r_m: float, r_deg: float,
) -> torch.Tensor:
    """Index of the mode nearest to the GT within r_m metres and r_deg degrees, else -1. [B]"""
    dist = (modes[..., :2].to(xy_gt) - xy_gt[:, None]).norm(dim=-1) / ppm
    yaw = modes[..., 2].to(yaw_gt) * (360.0 / num_rotations)
    dyaw = (yaw - yaw_gt[:, None]).abs() % 360
    dyaw = torch.minimum(dyaw, 360 - dyaw)
    ok = valid & (dist <= r_m) & (dyaw <= r_deg)
    dist = dist.masked_fill(~ok, float("inf"))
    t = dist.argmin(1)
    none = torch.isinf(dist.gather(1, t[:, None])[:, 0])
    return torch.where(none, torch.full_like(t, -1), t)


def reweight_log_probs(
    log_probs_img: torch.Tensor, basin_id: torch.Tensor, logp_modes: torch.Tensor,
    log_q: torch.Tensor, valid: torch.Tensor, tie_break: float = 1e-4,
) -> torch.Tensor:
    """Spec 4.4: inside basin k the image log-probs are shifted by log q_k - l_k (shape kept,
    ranking between basins set by the verifier); poses outside every basin are shifted so
    that they rank below all basins (l_1 -> min_k log q_k - 1). Re-normalised.
    Equal verifier scores are broken towards the lower mode index (tie_break nats per
    index), the same choice log_q.argmax makes, so the arg-max is always the selected mode."""
    log_q = log_q - tie_break * torch.arange(log_q.shape[1], device=log_q.device, dtype=log_q.dtype)
    shift_k = torch.where(valid, log_q - logp_modes, torch.zeros_like(log_q))
    min_q = log_q.masked_fill(~valid, float("inf")).min(1).values
    outside = (min_q - 1) - logp_modes[:, 0]
    idx = basin_id.clamp(min=0).flatten(1)
    shift = shift_k.gather(1, idx).reshape(basin_id.shape)
    shift = torch.where(basin_id >= 0, shift, outside[:, None, None, None].expand_as(shift))
    return log_softmax_spatial(log_probs_img + shift)


# ---------------------------------------------------------------------------
# caption -> descriptor verifier
# ---------------------------------------------------------------------------
def region_features(centres: torch.Tensor, num_freqs: int = 3) -> torch.Tensor:
    """[R, 2] (bearing rad, range) -> [R, 4 + 4 * num_freqs]."""
    phi, r = centres[:, 0], centres[:, 1]
    feats = [torch.cos(phi), torch.sin(phi), r, 1 - r]
    for k in range(num_freqs):
        f = 2.0**k
        feats += [torch.sin(f * phi), torch.cos(f * phi), torch.sin(f * math.pi * r), torch.cos(f * math.pi * r)]
    return torch.stack(feats, -1)


class CrossAttnLayer(nn.Module):
    """Pre-norm cross-attention + FFN, no self-attention (R queries attend to <= 64 tokens)."""

    def __init__(self, d: int, heads: int, ffn: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.ReLU(inplace=True), nn.Linear(ffn, d))

    def forward(self, q, mem, mem_mask):
        q = q + self.attn(self.n1(q), mem, mem, key_padding_mask=~mem_mask, need_weights=False)[0]
        return q + self.ffn(self.n2(q))


class RegionVerifier(nn.Module):
    """Learnable region queries (+ region position features) cross-attend to caption
    tokens and predict the OSM class logits of every region: [B, L, text_dim] -> [B, R, C]
    (softmax over C gives the fractions; log-softmax feeds the descriptor cross-entropy)."""

    def __init__(
        self, text_dim: int, centres: torch.Tensor, d_model: int = 256, layers: int = 2,
        heads: int = 8, ffn: int = 512, num_classes: int = 8,
    ):
        super().__init__()
        feats = region_features(centres)
        self.register_buffer("feats", feats, persistent=False)
        self.query = nn.Parameter(torch.randn(centres.shape[0], d_model) * 0.02)
        self.pos_proj = nn.Linear(feats.shape[-1], d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.layers = nn.ModuleList([CrossAttnLayer(d_model, heads, ffn) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = (self.query + self.pos_proj(self.feats))[None].expand(tokens.shape[0], -1, -1)
        mem = self.text_proj(tokens)
        for layer in self.layers:
            q = layer(q, mem, mask)
        return self.head(self.norm(q))
