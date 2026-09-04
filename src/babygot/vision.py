"""From-scratch Vision Transformer producing *patch-level* features.

BabyGOT's core claim is that grounding requires spatial structure.  Global
pooling (the LLaVA/BabyVLM connector) destroys that structure; we therefore keep
the full grid of patch tokens  z_v in R^{N x d}  (N = (H/P)^2) as the visual
representation that the optimal-transport head aligns to words.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import VisionConfig


class PatchEmbed(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.proj = nn.Conv2d(cfg.in_channels, cfg.width,
                              kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.n_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, cfg.width))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, d)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x + self.pos


class Block(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        d = cfg.width
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.heads, dropout=cfg.dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, int(d * cfg.mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d * cfg.mlp_ratio), d),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Minimal ViT.  Returns patch tokens (and optionally a CLS token)."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.patch = PatchEmbed(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.width)
        self.n_patches = self.patch.n_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, d)
        h = self.patch(x)
        for blk in self.blocks:
            if self.cfg.checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(blk, h, use_reentrant=False)
            else:
                h = blk(h)
        return self.norm(h)   # patch tokens only; spatial structure preserved


def patch_features(vit: ViT, x: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper returning normalised patch features."""
    return vit(x)
