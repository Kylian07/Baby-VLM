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
``ar``          **Headline.**  Autoregressive captioning (the objective BabyVLM-V2
                actually uses) with and without the referential-alignment
                auxiliary loss, swept across referential ambiguity.
``yusmith``     Fit the ideal observer's one free parameter (memory decay) to
                human accuracy in Yu & Smith (2007).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Running this as `python scripts/foo.py` puts scripts/ on sys.path, not the
# repo root, so `import gavagai` would fail. Make the script work regardless of
# how it is invoked or what the working directory is.
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from gavagai.sim import (
    ARConfig,
    EmbedConfig,
    LearnerConfig,
    ReferentialWorld,
    WorldConfig,
    run_ar_learner,
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
    family_spread=0.7,
    n_exemplars=8,
    within_spread=0.25,
)
# Calibrated so the captioning baseline sits near 0.64 in the clean regime:
# far from both the 0.025 chance floor and the ceiling, so conditions can
# separate in either direction.

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

    An unconstrained ideal observer solves Yu & Smith's task perfectly at every
    memory-decay rate, so forgetting is *not* the capacity limit that explains
    human performance.  The account that does is limited encoding: a learner
    registers only a few of the word-object pairs available on each trial.  We
    fit that single parameter, shared across all three conditions.
    """
    print("=== Yu & Smith (2007): fitting one limited-encoding parameter ===")
    rows = []
    for k in args.attend:
        preds, errs = {}, []
        for m, human in YU_SMITH_HUMAN.items():
            accs = [
                run_learner(
                    ReferentialWorld(WorldConfig(n_words=18, objects_per_trial=m, seed=s)),
                    LearnerConfig(attend_k=None if k <= 0 else k, rho=None),
                    n_trials=27,
                )["accuracy"]
                for s in range(args.seeds)
            ]
            preds[m] = float(np.mean(accs))
            errs.append((preds[m] - human) ** 2)
        rmse = float(np.sqrt(np.mean(errs)))
        rows.append(dict(attend_k=k, pred=preds, human=YU_SMITH_HUMAN, rmse=rmse))
        print(
            f"  attend_k={k:<4} "
            + "  ".join(f"{m}x{m}: {preds[m]:.3f} (human {YU_SMITH_HUMAN[m]:.3f})" for m in preds)
            + f"   RMSE={rmse:.3f}",
            flush=True,
        )
    best = min(rows, key=lambda r: r["rmse"])
    print(f"  best attend_k = {best['attend_k']} (RMSE {best['rmse']:.3f})")
    return rows


# Ambiguity regimes.  "realistic" reflects the properties Vong et al. report for
# caregiver speech in SAYCam: the named referent is visible only a minority of
# the time, most tokens are non-referential, and a handful of never-named
# objects (hands, floor, table) are in almost every frame.
AR_WORLDS = {
    "clean":     dict(null_rate=0.2, absent_ref_prob=0.0, n_background=3),
    "moderate":  dict(null_rate=0.5, absent_ref_prob=0.3, n_background=3),
    "realistic": dict(null_rate=0.8, absent_ref_prob=0.5, n_background=8),
}

AR_CONDITIONS = {
    "AR only (captioning)":        dict(aux_weight=0.0),
    "+ naive align (rho=0,no null)": dict(aux_weight=1.0, rho=0.0, use_null=False),
    "+ null bin only":             dict(aux_weight=1.0, rho=0.0, use_null=True),
    "+ null + ME (OURS)":          dict(aux_weight=1.0, rho=1.0, use_null=True),
    "+ null + balanced":           dict(aux_weight=1.0, rho=None, use_null=True),
}


def exp_ar(args):
    """Captioning objective +/- referential alignment, across ambiguity regimes.

    Readout is picture-vocabulary accuracy on *held-out* exemplars of each
    category, so it measures generalisation rather than recall of one vector.
    """
    print("=== autoregressive captioning +/- referential alignment ===")
    print(f"    corpus={args.corpus_size} episodes, {args.seeds} seeds, held-out exemplar readout")
    rows = []
    selected = args.regimes or list(AR_WORLDS)
    for wname, wkw in ((k, AR_WORLDS[k]) for k in selected):
        print(f"\n  [{wname}] {wkw}")
        for cname, ckw in AR_CONDITIONS.items():
            accs = []
            for s in range(args.seeds):
                world = ReferentialWorld(WorldConfig(seed=s, **(DEFAULT_WORLD | wkw)))
                res = run_ar_learner(
                    world,
                    ARConfig(steps=args.steps, seed=s, feat_noise=args.feat_noise, **ckw),
                    corpus=world.sample_corpus(args.corpus_size),
                )
                accs.append(res["accuracy"])
            rec = dict(regime=wname, condition=cname, acc_mean=float(np.mean(accs)),
                       acc_std=float(np.std(accs)), accs=accs)
            rows.append(rec)
            print(f"    {cname:32s} {rec['acc_mean']:.3f} +- {rec['acc_std']:.3f}", flush=True)
    return rows


EXPERIMENTS = {
    "ar": exp_ar,
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
    ap.add_argument("--regimes", nargs="+", default=None,
                    choices=sorted(AR_WORLDS), help="subset of ambiguity regimes to run")
    ap.add_argument("--corpus-size", type=int, default=500)
    ap.add_argument("--corpus-sizes", type=int, nargs="+", default=[100, 300, 1000, 3000])
    ap.add_argument("--feat-noise", type=float, default=0.35)
    ap.add_argument("--attend", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="objects encoded per trial; 0 means unlimited")
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
