#!/usr/bin/env python3
"""Print the word-slot coupling for one utterance, to show how it is built.

Rows are the *content words* of the utterance. Columns are *spatial regions of
the image* -- not words -- plus one null column meaning "refers to nothing
visible". This script makes both choices visible for a concrete input.

    python scripts/explain_coupling.py --text "look at that -- the red ball!"
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from gavagai.data.corpora import STOPWORDS, WordTokenizer, tokenize
from gavagai.models import ModelConfig
from gavagai.ot import referential_plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", default="oh look at that -- are you hungry? the red ball!")
    ap.add_argument("--slot-grid", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    cfg = ModelConfig(image_size=args.image_size, patch_size=args.patch,
                      slot_grid=args.slot_grid)

    # ---- ROWS ------------------------------------------------------------
    toks = tokenize(args.text)
    tok = WordTokenizer.build([args.text], min_count=1)
    kept_ids = tok.content_ids(args.text)
    kept = [tok.itos[i] for i in kept_ids]

    print(f'utterance: "{args.text}"\n')
    print(f"{'token':<12} {'kept?':<8} why")
    print("-" * 52)
    seen = set()
    for w in toks:
        if w in STOPWORDS:
            why, ok = "closed-class (function word)", False
        elif len(w) < 2:
            why, ok = "too short", False
        elif w in seen:
            why, ok = "already seen (deduplicated)", False
        else:
            why, ok = "content word -> ROW", True
            seen.add(w)
        print(f"{w:<12} {'YES' if ok else 'no':<8} {why}")

    # ---- COLUMNS ---------------------------------------------------------
    grid = cfg.image_size // cfg.patch_size
    g = cfg.slot_grid
    px = cfg.image_size // g
    n_slots = g * g
    print(f"\nimage {cfg.image_size}x{cfg.image_size} -> {grid}x{grid} patches"
          f" -> pooled to {g}x{g} = {n_slots} slots"
          f" (each slot = a {px}x{px} pixel region)")
    print(f"columns = {n_slots} slots + 1 NULL = {n_slots + 1}")
    print("\nthe columns are image regions, laid out spatially:")
    for r in range(g):
        print("   " + " ".join(f"[{r * g + c:>2}]" for c in range(g)))

    # ---- THE MATRIX ------------------------------------------------------
    n = len(kept)
    words = F.normalize(torch.randn(1, n, cfg.proj_dim), dim=-1)
    slots = F.normalize(torch.randn(1, n_slots, cfg.proj_dim), dim=-1)
    sim = torch.einsum("bnd,bmd->bnm", words, slots)

    plan, referential = referential_plan(
        sim, kappa=0.0, eps=args.eps, rho=args.rho, null_prior=0.5)

    print(f"\ncoupling P  ({n} words x {n_slots} slots + null),"
          f"  eps={args.eps}  rho={args.rho}")
    print("(untrained random weights -- the point is the SHAPE, not the values)\n")
    head = "word".ljust(12) + "".join(f"s{j:<4}" for j in range(n_slots)) + "NULL"
    print(head)
    print("-" * len(head))
    for i, w in enumerate(kept):
        row = "".join(f"{plan[0, i, j]:.2f} " for j in range(n_slots))
        null = 1.0 - float(referential[0, i])
        print(f"{w:<12}{row}{null:.2f}")
    print("\nreferential mass per word (1 = surely refers to something visible):")
    for i, w in enumerate(kept):
        print(f"  {w:<12} {float(referential[0, i]):.3f}")
    print("\nEach row sums to 1/N across all columns including NULL: every word is")
    print("accounted for, but 'accounted for' may mean 'refers to nothing here'.")


if __name__ == "__main__":
    main()
