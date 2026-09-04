#!/usr/bin/env python3
"""Zero-shot DevCV-Toolbox evaluation of a trained grounding model.

Both tasks are scored without any instruction tuning:

**Picture Vocabulary** -- for the queried word, score each candidate image by the
best-matching slot, and pick the argmax.

**Localization** -- take the word's argmax slot and map it back to its cell of
the patch grid.  Because slots are fixed spatial cells, this needs no bounding
box supervision at any point; it falls out of the transport plan.

    python scripts/evaluate.py --ckpt runs/gavagai/ckpt.pt \
        --tokenizer runs/gavagai/tokenizer.json --root data/Ego4D
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gavagai.data.corpora import UNK, WordTokenizer, tokenize
from gavagai.data.devcv import (
    evaluate_localization,
    evaluate_picture_vocabulary,
    find_tasks,
    load_task,
    slot_to_quadrant,
)
from gavagai.models import BabyVLM, ModelConfig


class Scorer:
    """Caches image encodings so each image is passed through the ViT once."""

    def __init__(self, model, tok, device, image_size):
        self.model, self.tok, self.device, self.image_size = model, tok, device, image_size
        self._cache: dict[Path, torch.Tensor] = {}
        self.grid = model.cfg.slot_grid

    def _image(self, path: Path) -> torch.Tensor:
        if path not in self._cache:
            from PIL import Image

            im = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
            a = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
            x = ((a - 0.5) / 0.5).unsqueeze(0).to(self.device)
            with torch.no_grad():
                self._cache[path] = self.model.encoder.encode_slots(x)[0]
        return self._cache[path]

    def _word(self, word: str) -> torch.Tensor | None:
        """Encode a possibly multi-word phrase; average over known tokens."""
        ids = [self.tok.stoi.get(w, UNK) for w in tokenize(word)]
        ids = [i for i in ids if i != UNK]
        if not ids:
            return None
        with torch.no_grad():
            v = self.model.word_embeddings_for_alignment(
                torch.tensor(ids, device=self.device).unsqueeze(0)
            )[0]
        return F.normalize(v.mean(0), dim=-1)

    def score(self, word: str, path: Path) -> float:
        w = self._word(word)
        if w is None:
            return 0.0
        return float((self._image(path) @ w).max())

    def quadrant(self, word: str, path: Path) -> str:
        w = self._word(word)
        if w is None:
            return "top left"
        return slot_to_quadrant(int((self._image(path) @ w).argmax()), self.grid, self.grid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--tasks", nargs="+", default=["picture_vocabulary", "localize"])
    ap.add_argument("--out", type=Path, default=Path("results/devcv_eval.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = WordTokenizer.load(args.tokenizer)
    state = torch.load(args.ckpt, map_location=device)
    saved = state.get("args", {})
    cfg = ModelConfig(image_size=saved.get("image_size", 128), vocab_size=len(tok))
    model = BabyVLM(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    available = find_tasks(args.root)
    if not available:
        raise SystemExit(f"no DevCV tasks found under {args.root}")
    print(f"  discovered: {', '.join(sorted(available))}")

    scorer = Scorer(model, tok, device, cfg.image_size)
    out = {}
    for task in args.tasks:
        try:
            items = load_task(args.root, task, args.split)
        except FileNotFoundError as e:
            print(f"  {task}: {e}")
            continue
        if task.startswith("local"):
            res = evaluate_localization(items, scorer.quadrant)
        else:
            res = evaluate_picture_vocabulary(items, scorer.score)
        out[task] = res
        print(f"  {task:22s} acc={res['accuracy']:.3f}  n={res['n']}  "
              f"chance={res['chance']:.3f}  skipped={res['n_skipped']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
