"""Spatially-ordered visual summary + token-wise gated fusion.

Two ideas are combined here:

1. **Spatial summary tokens.**  Instead of collapsing the image to one global
   vector (the LLaVA/BabyVLM connector — an information bottleneck identified by
   Ganescu et al., 2025), the patch grid is pooled into a *spatially ordered*
   ``grid x grid`` set of tokens (2-D adaptive average pooling).  Location is
   therefore a first-class part of the representation: token ``(r,c)`` summarises
   region ``(r,c)``, so the LM can read "left/right/top/bottom" and count
   directly — the exact abilities the global connector destroys (Localization,
   Spatial Details, Counting in DevCV Toolbox).

2. **Token-wise dynamic gating.**  A scalar gate gamma_t in (0,1) per word
   controls how strongly the word's representation absorbs visual evidence
   (Ganescu et al., 2025).  We show in the paper that gamma learns an
   interpretable content-word / function-word split without supervision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FusionConfig


class SpatialSummary(nn.Module):
    """Pools patch features to a fixed spatial grid: (B, N, d) -> (B, g*g, d).

    Row-major order, so token index k = r*g + c is the (r, c) region.
    """

    def __init__(self, cfg: FusionConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.grid = cfg.summary_grid

    def forward(self, z_v: torch.Tensor) -> torch.Tensor:
        B, N, d = z_v.shape
        n = int(round(N ** 0.5))
        x = z_v.view(B, n, n, d).permute(0, 3, 1, 2)          # (B, d, n, n)
        x = F.adaptive_avg_pool2d(x, (self.grid, self.grid))  # (B, d, g, g)
        return x.permute(0, 2, 3, 1).reshape(B, self.grid * self.grid, d)


class GatedFusion(nn.Module):
    """gamma-gated cross-attention from text tokens to spatial summary tokens.

    h_t  <-  h_t + gamma_t * MHCA(q = h_t, k = v = G)
    gamma_t = sigmoid(w_g^T h_t)      (per-token)
    """

    def __init__(self, cfg: FusionConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.summary = SpatialSummary(cfg, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.gate = nn.Linear(d_model, 1)
        nn.init.constant_(self.gate.bias, cfg.gate_bias)

    def forward(self, h_t: torch.Tensor, z_v: torch.Tensor):
        """h_t: (B, M, d) text embeddings; z_v: (B, N, d) patch features.

        Returns (h_fused, gamma).  When ``cfg.gate`` is False the cross-attention
        context is added *ungated* (gamma = 1) — the no_gate ablation.
        """
        G = self.summary(z_v)                       # (B, g*g, d)
        hq = self.norm(h_t)
        ctx, _ = self.attn(hq, G, G)                # (B, M, d)
        if self.cfg.gate:
            gamma = torch.sigmoid(self.gate(h_t))   # (B, M, 1)
            return h_t + gamma * ctx, gamma
        ones = torch.ones_like(ctx[..., :1])
        return h_t + ctx, ones
