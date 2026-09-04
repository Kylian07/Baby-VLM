"""From-scratch causal language model (a "tiny GPT").

We keep the LM small (a few million parameters) so that (i) it matches the
developmental-plausibility constraint of the BabyVLM workshop and (ii) it can be
pretrained *from scratch* on a single T4 within the workshop's budget.  The LM is
also the *generative* arm of BabyGOT's hybrid objective.

Convention: ``tok_emb`` returns token embeddings *without* positional encoding;
``forward_emb`` adds position once and runs the transformer.  This keeps the
gated-fusion path (model.py) free of double position-encoding bugs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import TextConfig


class CausalAttention(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        d = cfg.width
        assert d % cfg.heads == 0
        self.heads = cfg.heads
        self.head_dim = d // cfg.heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(mask[None, None, :, :] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, d)
        return self.out(y)


class CausalBlock(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        d = cfg.width
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalAttention(cfg)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, int(d * cfg.mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d * cfg.mlp_ratio), d),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.width)
        self.pos = nn.Embedding(cfg.max_len, cfg.width)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([CausalBlock(cfg) for _ in range(cfg.depth)])
        self.ln = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying

    def tok_emb(self, ids: torch.Tensor) -> torch.Tensor:
        """Token embeddings, no positional encoding.  (B, T, d)."""
        return self.drop(self.tok(ids))

    def forward_emb(self, x: torch.Tensor) -> torch.Tensor:
        """Add positions (once) and run the causal stack.  x: (B, T, d)."""
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + self.pos(pos[:, :T])
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        for blk in self.blocks:
            if self.cfg.checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, mask,
                                                      use_reentrant=False)
            else:
                x = blk(x, mask)
        return self.ln(x)

    def forward(self, ids: torch.Tensor,
                features: torch.Tensor | None = None) -> torch.Tensor:
        """Hidden states for token ids; ``features`` (visual tokens) are
        *prepended* and never supervised (BabyLLaVA-style baselines)."""
        x = self.tok_emb(ids)
        if features is not None:
            x = torch.cat([features, x], dim=1)          # (B, K+T, d)
        return self.forward_emb(x)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)
