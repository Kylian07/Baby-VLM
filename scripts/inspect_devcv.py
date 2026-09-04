#!/usr/bin/env python3
"""Report the actual shape of a DevCV release, so the audit can be matched to it.

The text-blind image-matching baseline only applies to items shaped "here is X,
which of (A)(B)(C) is the same X?", which it detects as `n_images ==
n_options + 1`.  When that heuristic does not fire, the audit reports `n/a` --
correct, but uninformative.  This script prints what the data actually looks
like so the heuristic can be corrected rather than guessed at.

    python scripts/inspect_devcv.py --root data/Ego4D
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from gavagai.data.devcv import find_tasks, load_source, target_word

_OPTION = re.compile(r"\(([A-F])\)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--examples", type=int, default=1, help="prompts to print per task")
    args = ap.parse_args()

    for task, sources in sorted(find_tasks(args.root).items()):
        items = []
        for s in sources:
            items.extend(load_source(s))
        if not items:
            continue

        n_img = Counter(it.n_images for it in items)
        n_opt = Counter(len(set(_OPTION.findall(it.prompt))) for it in items)
        answers = Counter(it.answer.strip()[:12] for it in items)
        parsed_word = sum(target_word(it.prompt) is not None for it in items)
        exists = sum(1 for it in items[:50] for p in it.images if p.exists())
        total_paths = sum(len(it.images) for it in items[:50])

        print("=" * 74)
        print(f"{task}   ({len(items)} items)")
        print(f"  n_images      : {dict(sorted(n_img.items()))}")
        print(f"  n_options     : {dict(sorted(n_opt.items()))}")
        print(f"  answers (top6): {dict(answers.most_common(6))}")
        print(f"  target_word parsed: {parsed_word}/{len(items)}")
        print(f"  image paths exist : {exists}/{total_paths} (first 50 items)")
        for it in items[: args.examples]:
            print(f"  --- example id={it.item_id} gold={it.answer!r} ---")
            print(f"      prompt: {it.prompt[:300]}")
            for p in it.images[:6]:
                print(f"      image : {p.name}  exists={p.exists()}")


if __name__ == "__main__":
    main()
