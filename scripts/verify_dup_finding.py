#!/usr/bin/env python3
"""Independently verify the duplicate-image finding, using only the stdlib.

A result of 1.00 with a zero-width confidence interval is usually a bug in the
measuring instrument, so this script deliberately shares NO code with
``gavagai`` -- no loader, no resolver, no path handling. It reads the released
JSON directly and checks two claims:

1. spatialdetails: in each item, is exactly one option string identical to the
   query string, and is that option the gold answer?
2. leftright: are all image strings within an item identical, making the item
   unanswerable from the released files?

    python scripts/verify_dup_finding.py --root data/Ego4D
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

LETTERS = "ABCDEF"


def records(path: Path):
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("items") or []
    return payload


def analyse(path: Path) -> dict:
    n = dup_is_gold = dup_exactly_one = all_identical = other = 0
    gold_positions = Counter()

    for rec in records(path):
        imgs = rec.get("image") or []
        if isinstance(imgs, str):
            imgs = [imgs]
        convs = rec.get("conversations", [])
        human = next((c["value"] for c in convs if c.get("from") == "human"), "")
        gpt = next((c["value"] for c in convs if c.get("from") == "gpt"), "").strip()
        n_opts = len(set(re.findall(r"\(([A-F])\)", human)))
        if len(imgs) != n_opts + 1 or gpt not in LETTERS:
            continue

        n += 1
        gold = LETTERS.index(gpt)
        gold_positions[gpt] += 1
        query, options = imgs[0], imgs[1:]
        same = [i for i, o in enumerate(options) if o == query]

        if len(same) == len(options):
            all_identical += 1
        elif len(same) == 1:
            dup_exactly_one += 1
            dup_is_gold += int(same[0] == gold)
        else:
            other += 1

    return {
        "items_scored": n,
        "all_images_identical": all_identical,
        "exactly_one_duplicate": dup_exactly_one,
        "duplicate_is_the_gold_answer": dup_is_gold,
        "neither": other,
        "gold_letter_spread": dict(sorted(gold_positions.items())),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()

    files = [p for p in sorted(args.root.rglob("*.json"))
             if not any(part.startswith(".") for part in p.relative_to(args.root).parts[:-1])]
    if not files:
        raise SystemExit(f"no json files under {args.root}")

    for path in files:
        try:
            r = analyse(path)
        except Exception as e:
            print(f"{path.name}: unreadable ({type(e).__name__})")
            continue
        if not r["items_scored"]:
            continue
        print(f"\n=== {path.name} ===")
        for k, v in r.items():
            print(f"  {k:32s}: {v}")
        n, dup, hit = r["items_scored"], r["exactly_one_duplicate"], r["duplicate_is_the_gold_answer"]
        if dup:
            print(f"  --> the duplicate is the gold answer in {hit}/{dup} = {hit / dup:.4f}")
        if r["all_images_identical"]:
            frac = r["all_images_identical"] / n
            print(f"  --> {frac:.1%} of items list the SAME file for query and every option")


if __name__ == "__main__":
    main()
