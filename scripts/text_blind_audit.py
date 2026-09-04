#!/usr/bin/env python3
"""Audit DevCV-Toolbox tasks for text-blind solvability.

A multiple-choice vision-language benchmark measures grounding only to the
extent that its items cannot be answered *without reading the question*.  This
script measures how far each task can be pushed by baselines that never look at
the text:

``match``
    For items shaped "here is X, which of (A)(B)(C) is the same X?", pick the
    option most similar to the query image, using a deliberately weak feature
    (32x32 luminance + RGB histogram).  No training, no language, no network.

``position``
    Always answer the most frequent gold letter for that task.  Catches
    answer-position imbalance.

``random``
    Uniform over the options; the nominal chance floor.

Why this matters: the published baby model scores 96.4 on Left/Right and 92.8
on Spatial Details -- at or above the human ceiling on the first -- while
sitting at chance on Picture Vocabulary.  If the high-scoring tasks are largely
text-blind-solvable, those numbers are not evidence of grounding, and the
apparent dissociation is partly an artefact of what the tasks measure.

Run against the public Ego4D variant for the headline numbers:

    hf download wsashawn/devcv_toolbox_ego4d --repo-type dataset --local-dir data/Ego4D
    python scripts/text_blind_audit.py --roots data/Ego4D
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

# Running this as `python scripts/foo.py` puts scripts/ on sys.path, not the
# repo root, so `import gavagai` would fail. Make the script work regardless of
# how it is invoked or what the working directory is.
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from gavagai.data.devcv import find_tasks, load_source

_OPTION = re.compile(r"\(([A-F])\)")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- the right CI for small n, which is what we have."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def image_feature(path: Path, size: int = 32, bins: int = 8) -> np.ndarray:
    """Deliberately weak descriptor: normalised luminance + colour histogram."""
    from PIL import Image

    im = Image.open(path).convert("RGB").resize((size, size))
    a = np.asarray(im, dtype=np.float32) / 255.0
    gray = a.mean(2).ravel()
    gray = (gray - gray.mean()) / (gray.std() + 1e-6)
    gray /= np.linalg.norm(gray) + 1e-9
    hist = np.concatenate(
        [np.histogram(a[:, :, c], bins=bins, range=(0, 1))[0] for c in range(3)]
    ).astype(np.float32)
    hist /= np.linalg.norm(hist) + 1e-9
    return np.concatenate([gray, 3.0 * hist])


def n_options(prompt: str) -> int:
    return len(set(_OPTION.findall(prompt)))


def discover(roots: list[Path]) -> dict[str, list]:
    """Find every DevCV task under the given roots, whichever layout they use.

    Handles the flat Hugging Face release (``<task>_test.json`` beside a shared
    ``images/``) as well as the nested website samples.
    """
    found: dict[str, list] = {}
    for root in roots:
        for task, sources in find_tasks(root).items():
            found.setdefault(task, []).extend(sources)
    return found


def audit_task(entries) -> dict:
    items = []
    for src in entries:
        try:
            items.extend(load_source(src))
        except (FileNotFoundError, ValueError):
            continue

    match_ok = match_n = 0
    golds, opt_counts = [], []
    for it in items:
        k = n_options(it.prompt)
        gold = it.answer_index
        if gold is None or k < 2:
            continue
        golds.append(gold)
        opt_counts.append(k)
        # Matching baseline applies when there is a query image plus one image
        # per option: images = [query, optA, optB, ...].
        if it.n_images == k + 1:
            try:
                feats = [image_feature(p) for p in it.images]
            except Exception:
                continue
            q = feats[0]
            sims = [float(q @ f) for f in feats[1:]]
            match_ok += int(int(np.argmax(sims)) == gold)
            match_n += 1

    if not golds:
        return {}
    counts = Counter(golds)
    pos_ok = counts.most_common(1)[0][1]
    chance = float(np.mean([1.0 / k for k in opt_counts]))
    out = {
        "n_items": len(golds),
        "chance": chance,
        "position_acc": pos_ok / len(golds),
        "position_ci": wilson(pos_ok, len(golds)),
    }
    if match_n:
        out |= {
            "match_acc": match_ok / match_n,
            "match_n": match_n,
            "match_ci": wilson(match_ok, match_n),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("results/text_blind_audit.json"))
    args = ap.parse_args()

    found = discover(args.roots)
    if not found:
        raise SystemExit(
            f"no DevCV tasks discovered under {[str(r) for r in args.roots]}.\n"
            "Expected either <root>/<task>_test.json beside an images/ directory "
            "(the Hugging Face release) or <root>/[<split>/]<task>/data.json."
        )
    print(f"discovered {len(found)} task(s): {', '.join(sorted(found))}\n")
    print(f"{'task':24s} {'n':>4} {'chance':>7} {'position':>18} {'image-match':>20}")
    print("-" * 78)
    rows = {}
    for task in sorted(found):
        r = audit_task(found[task])
        if not r:
            continue
        rows[task] = r
        pos = f"{r['position_acc']:.2f} [{r['position_ci'][0]:.2f},{r['position_ci'][1]:.2f}]"
        mat = (
            f"{r['match_acc']:.2f} [{r['match_ci'][0]:.2f},{r['match_ci'][1]:.2f}] n={r['match_n']}"
            if "match_acc" in r
            else "n/a"
        )
        print(f"{task:24s} {r['n_items']:>4} {r['chance']:>7.2f} {pos:>18} {mat:>20}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")
    print(
        "\nNOTE: on the bundled website samples n is tiny (single digits per task).\n"
        "Wilson intervals are reported for exactly that reason.  Run against the\n"
        "full public Ego4D release before quoting any of these numbers."
    )


if __name__ == "__main__":
    main()
