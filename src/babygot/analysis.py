"""Interpretability analyses that make BabyGOT's two mechanisms *visible*:

1. **Referential grounding map.**  For a caption word w and an image, the OT
   coupling P* gives  P*_{i,w}  = P(region i | word w), the probability that word
   w refers to region i.  Reshaping it onto the patch grid recovers the model's
   "pointing" behaviour — directly comparable to an infant's gaze / pointing in
   the Looking-While-Listening paradigm.  We report the centre of mass of that
   map against the true object cell (a localisation score).

2. **Token-wise gate.**  gamma_t in (0,1) says how much word t "looks at" the
   image.  We split words into content words (nouns, adjectives, numerals) and
   function words and check that gamma separates them without supervision
   (cf. Ganescu et al., 2025).

Both are cheap, single-forward-pass quantities already computed by the model.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch

from .data import GRID, SHAPE_NOUN, SHAPES, COLORS, SIZES, NUMBERS, POS_LABEL, ROW_LABEL
from .model import BabyGOT
from .tokenizer import Tokenizer

CONTENT = (set(SHAPES) | set(SHAPE_NOUN.values()) | set(SHAPE_NOUN.values()) |
           set(COLORS) | set(SIZES) | set(NUMBERS) | set(POS_LABEL) | set(ROW_LABEL) |
           {n + "s" for n in SHAPE_NOUN.values()})


def _patch_grid(n_patches: int, image_size: int, patch_size: int) -> torch.Tensor:
    n = image_size // patch_size
    ys = (torch.arange(n_patches) // n).float() + 0.5
    xs = (torch.arange(n_patches) % n).float() + 0.5
    return torch.stack([ys, xs], dim=1)          # (N, 2) in patch units


@torch.no_grad()
def word_region_map(model: BabyGOT, image: torch.Tensor, word: str,
                    tokenizer: Tokenizer) -> torch.Tensor:
    """P(region i | word) shaped as a (sqrt N, sqrt N) grid.  Returns (Hp, Wp)."""
    cfg = model.cfg
    dev = next(model.parameters()).device
    img = image.unsqueeze(0).to(dev)
    z_v = model.encode_image(img)                      # (1, N, d)
    ids = torch.tensor([tokenizer.encode(word, add_bos=True, add_eos=True)],
                       device=dev)
    z_t = model.text.tok_emb(ids)                       # (1, M, d)
    P = model.align(z_v, z_t)["P"][0]                   # (N, M)
    wid = (ids[0] == tokenizer.vocab.get(word, -1)).nonzero()
    if len(wid) == 0:
        return torch.zeros(1, 1)
    col = P[:, wid[0, 0]]                              # (N,)
    col = col / (col.sum() + 1e-9)
    n = cfg.vision.image_size // cfg.vision.patch_size
    return col.view(n, n).cpu()


@torch.no_grad()
def localization_error(model: BabyGOT, image: torch.Tensor, word: str,
                       cell, tokenizer: Tokenizer) -> float:
    """Mean distance (in patch units) between the grounding map's centre of mass
    and the true object cell centre.  Lower is better (better pointing)."""
    m = word_region_map(model, image, word, tokenizer)   # (Hp, Wp)
    Hp, Wp = m.shape
    ys = torch.arange(Hp).float().view(-1, 1)
    xs = torch.arange(Wp).float().view(1, -1)
    com_y = (m * ys).sum() / (m.sum() + 1e-9)
    com_x = (m * xs).sum() / (m.sum() + 1e-9)
    n = Hp
    step = n / GRID
    true_y = (cell[0] + 0.5) * step
    true_x = (cell[1] + 0.5) * step
    return float(((com_y - true_y) ** 2 + (com_x - true_x) ** 2) ** 0.5)


@torch.no_grad()
def gate_split(model: BabyGOT, image: torch.Tensor, text: str,
               tokenizer: Tokenizer) -> Dict[str, float]:
    """Mean gate value on content vs function words of a caption."""
    if not model.gated:
        return {"content": 0.0, "function": 0.0}
    dev = next(model.parameters()).device
    img = image.unsqueeze(0).to(dev)
    ids = torch.tensor([tokenizer.encode(text, add_bos=True, add_eos=True)],
                       device=dev)
    z_v = model.encode_image(img)
    tok = model.text.tok_emb(ids)
    _, gamma = model.fusion(tok, z_v)                    # (1, M, 1)
    words = text.lower().split()
    content, function, n_c, n_f = 0.0, 0.0, 0, 0
    for i, w in enumerate(words[: gamma.shape[1]]):
        g = float(gamma[0, i, 0])
        if w in CONTENT:
            content += g
            n_c += 1
        else:
            function += g
            n_f += 1
    return {"content": content / max(n_c, 1),
            "function": function / max(n_f, 1)}


def _single_object_scenes(n: int, seed: int, image_size: int):
    """Deterministic single-object naming scenes for a clean pointing test."""
    import random
    from .data import render_scene, SHAPES, COLORS, SHAPE_NOUN

    rng = random.Random(seed)
    out = []
    for _ in range(n):
        shape, color = rng.choice(SHAPES), rng.choice(COLORS)
        cell = (rng.randrange(GRID), rng.randrange(GRID))
        img = render_scene([{"shape": shape, "color": color, "size": "big",
                             "cell": cell}], image_size)
        caption = f"a {color} {SHAPE_NOUN[shape]}"
        out.append({"image": img, "caption": caption, "cell": cell,
                    "word": color})
    return out


def analyze(model: BabyGOT, tokenizer: Tokenizer, items=None,
            n_scenes: int = 64, image_size: int = 96, seed: int = 0) -> Dict:
    """Run the two interpretability analyses.

    If ``items`` is None we generate clean single-object naming scenes (so the
    pointing metric is not confounded by multi-object captions).
    """
    model.eval()
    if items is None:
        items = _single_object_scenes(n_scenes, seed, image_size)
    errs, gate_c, gate_f = [], [], []
    for it in items:
        word = it.get("word") or it["facts"]["target"]["color"]
        cell = it.get("cell") or it["facts"]["target"]["cell"]
        errs.append(localization_error(model, it["image"], word, cell, tokenizer))
        g = gate_split(model, it["image"], it["caption"], tokenizer)
        gate_c.append(g["content"])
        gate_f.append(g["function"])
    n = max(len(errs), 1)
    return {"mean_localization_error_patch": sum(errs) / n,
            "gate_content": sum(gate_c) / n,
            "gate_function": sum(gate_f) / n}
