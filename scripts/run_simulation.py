#!/usr/bin/env python3
"""Controlled cross-situational experiments (CPU-only, minutes).

These runs isolate the mechanism from every confound of real vision, and they
are cheap enough to repeat with many seeds -- which is what makes them the
right place to establish that the effect is real before spending GPU hours.

Experiments
-----------
``efficiency``  Accuracy vs. size of a *finite* corpus.  The headline result:
                mutual exclusivity buys sample efficiency.
``ladder``      Ablation over the ingredients at one corpus size.
``rho``         Sweep the mutual-exclusivity strength continuously from the
                region-word contrastive baseline (rho=0) to balanced OT.
``yusmith``     Fit the ideal observer's one free parameter (memory decay) to
                human accuracy in Yu & Smith (2007).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gavagai.sim import (
    EmbedConfig,
    LearnerConfig,
    ReferentialWorld,
    WorldConfig,
    run_embedding_learner,
    run_learner,
)

# Yu & Smith (2007), Psychological Science 18(5):414-420.  Proportion of the 18
# word-referent pairs learned after 27 trials, by within-trial ambiguity.
YU_SMITH_HUMAN = {2: 0.889, 3: 0.778, 4: 0.556}

# Default world: 40 nameable objects, 3 never-named background objects present
# in almost every scene, half of all heard words non-referential, Zipfian
# object frequencies, and visually confusable object families.
DEFAULT_WORLD = dict(
    n_words=40,
    objects_per_trial=4,
    n_background=3,
    background_prob=0.95,
    background_name_prob=0.02,
    null_rate=0.5,
    freq_skew=1.0,
    n_filler_words=30,
    n_feature_families=10,
    family_spread=0.4,
)

CONDITIONS = {
    "A_region_word_contrastive": dict(rho=0.0, use_null=False),
    "B_plus_null_bin": dict(rho=0.0, use_null=True),
    "C_plus_mutual_exclusivity": dict(rho=1.0, use_null=True),
    "D_balanced": dict(rho=None, use_null=True),
}


def _run_grid(corpus_sizes, conditions, seeds, steps, world_kw, embed_kw):
    out = []
    for size in corpus_sizes:
        for name, cond in conditions.items():
            accs, hubs = [], []
            for s in seeds:
                world = ReferentialWorld(WorldConfig(seed=s, **world_kw))
                corpus = world.sample_corpus(size) if size else None
                res = run_embedding_learner(
                    world, EmbedConfig(steps=steps, seed=s, **cond, **embed_kw), corpus=corpus
                )
                accs.append(res["accuracy"])
                hubs.append(res["hub_rate"])
            rec = dict(
                corpus_size=size,
                condition=name,
                acc_mean=float(np.mean(accs)),
                acc_std=float(np.std(accs)),
                hub_mean=float(np.mean(hubs)),
                accs=accs,
            )
            out.append(rec)
            print(
                f"  n={size!s:>6}  {name:28s} acc={rec['acc_mean']:.3f}"
                f" +-{rec['acc_std']:.3f}  hub={rec['hub_mean']:.3f}",
                flush=True,
            )
    return out


def exp_efficiency(args):
    print("=== sample efficiency: accuracy vs finite corpus size ===")
    return _run_grid(
        args.corpus_sizes, CONDITIONS, range(args.seeds), args.steps,
        DEFAULT_WORLD, dict(feat_noise=args.feat_noise),
    )


def exp_ladder(args):
    print(f"=== ablation ladder at corpus size {args.corpus_size} ===")
    return _run_grid(
        [args.corpus_size], CONDITIONS, range(args.seeds), args.steps,
        DEFAULT_WORLD, dict(feat_noise=args.feat_noise),
    )


def exp_rho(args):
    print(f"=== rho sweep at corpus size {args.corpus_size} ===")
    conds = {f"rho={r}": dict(rho=r, use_null=True) for r in [0.0, 0.02, 0.05, 0.1, 0.3, 1.0, 3.0, None]}
    return _run_grid(
        [args.corpus_size], conds, range(args.seeds), args.steps,
        DEFAULT_WORLD, dict(feat_noise=args.feat_noise),
    )


def exp_yusmith(args):
    """Ideal observer vs. humans on the original design, with one free parameter.

    An unconstrained ideal observer solves Yu & Smith's task perfectly, so the
    interesting question is not whether it can, but what capacity limit has to
    be assumed to reproduce the human accuracy *ordering*.  We fit a single
    memory-decay parameter shared across all three conditions.
    """
    print("=== Yu & Smith (2007) replication: fitting one memory-decay parameter ===")
    rows = []
    for gamma in args.gammas:
        preds, errs = {}, []
        for m, human in YU_SMITH_HUMAN.items():
            accs = [
                run_learner(
                    ReferentialWorld(WorldConfig(n_words=18, objects_per_trial=m, seed=s)),
                    LearnerConfig(gamma=gamma, rho=None),
                    n_trials=27,
                )["accuracy"]
                for s in range(args.seeds)
            ]
            preds[m] = float(np.mean(accs))
            errs.append((preds[m] - human) ** 2)
        rmse = float(np.sqrt(np.mean(errs)))
        rows.append(dict(gamma=gamma, pred=preds, human=YU_SMITH_HUMAN, rmse=rmse))
        print(
            f"  gamma={gamma:<6} "
            + "  ".join(f"{m}x{m}: {preds[m]:.3f} (human {YU_SMITH_HUMAN[m]:.3f})" for m in preds)
            + f"   RMSE={rmse:.3f}",
            flush=True,
        )
    best = min(rows, key=lambda r: r["rmse"])
    print(f"  best gamma = {best['gamma']} (RMSE {best['rmse']:.3f})")
    return rows


EXPERIMENTS = {
    "efficiency": exp_efficiency,
    "ladder": exp_ladder,
    "rho": exp_rho,
    "yusmith": exp_yusmith,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=sorted(EXPERIMENTS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--corpus-size", type=int, default=500)
    ap.add_argument("--corpus-sizes", type=int, nargs="+", default=[100, 300, 1000, 3000])
    ap.add_argument("--feat-noise", type=float, default=0.35)
    ap.add_argument("--gammas", type=float, nargs="+", default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    t0 = time.time()
    rows = EXPERIMENTS[args.experiment](args)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"sim_{args.experiment}.json"
    path.write_text(json.dumps({"args": vars(args) | {"out": str(args.out)}, "rows": rows}, indent=2))
    print(f"\nwrote {path}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
