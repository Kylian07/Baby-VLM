"""Compact vision-language grounding model sized for a Kaggle T4.

Nothing here is novel; it exists so that the objective in ``gavagai.losses`` can
be trained end to end at infant scale.  Deliberate choices:

* **Trained from scratch.**  No ImageNet or CLIP initialisation, since the point
  is what can be learned from an infant-sized budget.
* **Grid slot pooling by default.**  Slots are fixed spatial cells of the patch
  grid.  This is the cheapest option, and it makes the localization readout
  exact: a slot maps back to a known image region, so ``word -> argmax slot ->
  quadrant`` needs no box supervision.
* **Word-level vocabulary.**  Child-directed speech has a small vocabulary, and
  a word-level table keeps the lexicon in ``gavagai.lexicon`` indexable.

Default config (ViT-S/16 at 128px, 64 patches, 16 slots) is ~22M parameters and
trains comfortably in fp16 within a 16GB T4.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    image_size: int = 128
    patch_size: int = 16
    width: int = 384
    depth: int = 8
    heads: int = 6
    proj_dim: int = 256
    vocab_size: int = 8192
    text_width: int = 256
    slot_grid: int = 4
    """Slots per side.  With a 8x8 patch grid, slot_grid=4 gives 16 slots of 2x2
    patches each."""
    pool: str = "grid"
    dropout: float = 0.0


class _Block(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.n1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class VisionEncoder(nn.Module):
    """Plain ViT returning patch tokens plus a pooled global token."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.grid = cfg.image_size // cfg.patch_size
        self.patch = nn.Conv2d(3, cfg.width, cfg.patch_size, cfg.patch_size)
        self.pos = nn.Parameter(torch.zeros(1, self.grid**2, cfg.width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(_Block(cfg.width, cfg.heads, cfg.dropout) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch(images).flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class SlotPooler(nn.Module):
    """Reduce a patch grid to ``slot_grid**2`` spatial slots."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, tokens: torch.Tensor, grid: int) -> torch.Tensor:
        if self.cfg.pool == "none":
            return tokens
        b, p, d = tokens.shape
        g = self.cfg.slot_grid
        x = tokens.transpose(1, 2).reshape(b, d, grid, grid)
        x = F.adaptive_avg_pool2d(x, (g, g))
        return x.flatten(2).transpose(1, 2)

    @property
    def n_slots(self) -> int:
        return self.cfg.slot_grid**2

    def slot_grid_shape(self) -> tuple[int, int]:
        return self.cfg.slot_grid, self.cfg.slot_grid


class WordEncoder(nn.Module):
    """Word-level embedding table with a small contextual encoder."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.text_width)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.enc = nn.Sequential(
            nn.LayerNorm(cfg.text_width),
            nn.Linear(cfg.text_width, cfg.text_width),
            nn.GELU(),
        )

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        return self.enc(self.emb(word_ids))


class GroundingModel(nn.Module):
    """Vision encoder + slot pooling + word encoder projected to a shared sphere."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.vision = VisionEncoder(cfg)
        self.pool = SlotPooler(cfg)
        self.text = WordEncoder(cfg)
        self.vis_proj = nn.Linear(cfg.width, cfg.proj_dim)
        self.txt_proj = nn.Linear(cfg.text_width, cfg.proj_dim)
        # kappa is a *hyper-parameter*, not a learnable weight: it is only ever
        # read inside the no-grad E-step, so it receives no gradient (this is
        # Danskin's theorem doing its job, not a bug).  Sweep it instead.
        self.register_buffer("kappa", torch.tensor(0.0))
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / 0.07).log())

    def encode_slots(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.vision(images)
        slots = self.pool(tokens, self.vision.grid)
        return F.normalize(self.vis_proj(slots), dim=-1)

    def encode_global(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.vision(images)
        return F.normalize(self.vis_proj(tokens.mean(1)), dim=-1)

    def encode_words(self, word_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.txt_proj(self.text(word_ids)), dim=-1)

    @property
    def n_slots(self) -> int:
        return self.pool.n_slots

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CaptionDecoder(nn.Module):
    """Small autoregressive decoder conditioned on image slots as a prefix.

    This mirrors the LLaVA-style recipe BabyVLM-V2 actually uses -- visual
    tokens are prepended and the utterance is predicted left to right -- at a
    size that trains from scratch on a T4.  It is deliberately the *baseline*:
    the point of the experiment is that this objective alone supplies no
    pressure to bind a particular word to a particular region.
    """

    def __init__(self, cfg: ModelConfig, width: int = 256, depth: int = 4, heads: int = 4,
                 max_len: int = 32):
        super().__init__()
        self.width = width
        self.max_len = max_len
        self.tok = nn.Embedding(cfg.vocab_size, width)
        self.pos = nn.Parameter(torch.zeros(1, max_len + cfg.slot_grid**2, width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.normal_(self.tok.weight, std=0.02)
        self.connector = nn.Sequential(
            nn.Linear(cfg.proj_dim, width), nn.GELU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList(_Block(width, heads, cfg.dropout) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied

    def forward(self, slots: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Return logits over the utterance positions.

        Args:
            slots: ``(B, M, proj_dim)`` visual prefix.
            tokens: ``(B, L)`` utterance token ids.
        """
        prefix = self.connector(slots)
        x = torch.cat([prefix, self.tok(tokens)], dim=1)
        x = x + self.pos[:, : x.shape[1]]

        n = x.shape[1]
        m = slots.shape[1]
        # Causal over text; the visual prefix is fully visible to everything.
        mask = torch.full((n, n), float("-inf"), device=x.device).triu(1)
        mask[:, :m] = 0.0
        for blk in self.blocks:
            h = blk.n1(x)
            x = x + blk.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
            x = x + blk.mlp(blk.n2(x))
        return self.head(self.norm(x[:, m:]))


class BabyVLM(nn.Module):
    """Grounding encoder + caption decoder: the full model used in `scripts/train.py`."""

    def __init__(self, cfg: ModelConfig, decoder: bool = True):
        super().__init__()
        self.cfg = cfg
        self.encoder = GroundingModel(cfg)
        self.decoder = CaptionDecoder(cfg) if decoder else None
        self.align_proj = (
            nn.Linear(self.decoder.width, cfg.proj_dim) if decoder else None
        )

    def word_embeddings_for_alignment(self, word_ids: torch.Tensor) -> torch.Tensor:
        """Word vectors used by the alignment loss.

        When a decoder is present these are its (tied) token embeddings, so the
        auxiliary loss shapes exactly the parameters the captioning objective
        already owns and the two objectives cannot be trivially decoupled.
        """
        if self.decoder is None:
            return self.encoder.encode_words(word_ids)
        return F.normalize(self.align_proj(self.decoder.tok(word_ids)), dim=-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
