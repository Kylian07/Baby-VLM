"""Command-line entry point.

Examples
--------
    # quick smoke test (CPU, ~1 min)
    python -m babygot.run --method babygot --tiny

    # full run on a Kaggle T4
    python -m babygot.run --method babygot --steps 4000 --n-train 4000 --n-eval 200

    # the paper's ablation table (every method, same budget & seed)
    python -m babygot.run --all --steps 2000 --n-eval 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import Config, make_config
from .train import train

METHODS = ["babygot", "global_clip", "babyllava", "no_gate", "no_ot"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="BabyGOT: grounded optimal-transport VLM")
    p.add_argument("--method", choices=METHODS, default="babygot")
    p.add_argument("--all", action="store_true",
                   help="run every method and print a comparison table")
    p.add_argument("--tiny", action="store_true", help="tiny config (smoke test)")
    p.add_argument("--small", action="store_true",
                   help="mid-size config (fast but meaningful run)")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default="runs")
    return p.parse_args(argv)


def run_one(method: str, args) -> dict:
    cfg = make_config(method, tiny=args.tiny, small=args.small)
    if args.steps is not None:
        cfg.train.steps = args.steps
    if args.n_train is not None:
        cfg.n_train_scenes = args.n_train
    if args.n_eval is not None:
        cfg.n_eval_scenes = args.n_eval
    cfg.seed = args.seed
    _, _, res = train(cfg, save_dir=args.save_dir)
    return res


def main(argv=None):
    args = parse_args(argv)
    if args.all:
        table = {}
        for m in METHODS:
            print(f"\n===== method: {m} =====")
            res = run_one(m, args)
            table[m] = res
        print("\n===== comparison =====")
        keys = [k for k in table[METHODS[0]]
                if k not in ("overall_choice_acc", "analysis")]
        header = "method".ljust(12) + "".join(k.ljust(14) for k in keys)
        print(header)
        for m in METHODS:
            row = m.ljust(12)
            for k in keys:
                v = table[m].get(k, {})
                val = v.get("acc", v.get("f1", 0.0))
                row += f"{val:<14.3f}"
            print(row)
        print("\ninterpretability (BabyGOT family only):")
        for m in METHODS:
            a = table[m].get("analysis")
            if a:
                print(f"  {m:12s} loc_err={a['mean_localization_error_patch']:.2f}"
                      f" gate_content={a['gate_content']:.3f}"
                      f" gate_function={a['gate_function']:.3f}")
        return 0
    res = run_one(args.method, args)
    print("\n[result]", args.method, json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
