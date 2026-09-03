#!/usr/bin/env python3
"""Render figures from whatever result JSONs are present.

Every figure is generated from a file written by another script, so nothing here
can quietly disagree with the numbers actually measured.  Missing inputs are
skipped with a message rather than fabricated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#4C6EF5", "#F76707", "#0CA678", "#AE3EC9", "#868E96"]


def fig_ar_ablation(results: Path, out: Path) -> bool:
    src = results / "sim_ar.json"
    if not src.exists():
        return False
    rows = json.loads(src.read_text())["rows"]
    regimes = list(dict.fromkeys(r["regime"] for r in rows))
    conds = list(dict.fromkeys(r["condition"] for r in rows))

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    width = 0.8 / len(conds)
    for i, cond in enumerate(conds):
        vals = [next((r["acc_mean"] for r in rows if r["regime"] == g and r["condition"] == cond), 0) for g in regimes]
        errs = [next((r["acc_std"] for r in rows if r["regime"] == g and r["condition"] == cond), 0) for g in regimes]
        xs = [j + i * width - 0.4 + width / 2 for j in range(len(regimes))]
        ax.bar(xs, vals, width * 0.92, yerr=errs, capsize=3, label=cond, color=PALETTE[i % len(PALETTE)])
    ax.axhline(1 / 40, ls="--", c="0.4", lw=1)
    ax.text(len(regimes) - 0.55, 1 / 40 + 0.008, "chance", fontsize=8, color="0.35")
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels([g.capitalize() for g in regimes])
    ax.set_xlabel("referential ambiguity regime")
    ax.set_ylabel("picture-vocabulary accuracy\n(held-out exemplars)")
    ax.set_title("Captioning objective ± referential alignment", loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "ar_ablation.png", dpi=180)
    plt.close(fig)
    return True


def fig_text_blind(results: Path, out: Path) -> bool:
    src = results / "text_blind_audit.json"
    if not src.exists():
        return False
    rows = json.loads(src.read_text())
    tasks = [t for t, r in rows.items() if "match_acc" in r]
    if not tasks:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(tasks) + 2))
    ys = range(len(tasks))
    accs = [rows[t]["match_acc"] for t in tasks]
    los = [rows[t]["match_acc"] - rows[t]["match_ci"][0] for t in tasks]
    his = [rows[t]["match_ci"][1] - rows[t]["match_acc"] for t in tasks]
    ax.barh(list(ys), accs, xerr=[los, his], capsize=4, color=PALETTE[1], height=0.55)
    for y, t in zip(ys, tasks):
        ax.plot([rows[t]["chance"]], [y], marker="|", ms=14, c="0.25")
    ax.set_yticks(list(ys))
    ax.set_yticklabels(tasks)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("accuracy of a training-free matcher that never reads the prompt")
    ax.set_title("Text-blind solvability of DevCV-Toolbox tasks", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "text_blind_audit.png", dpi=180)
    plt.close(fig)
    return True


def fig_efficiency(results: Path, out: Path) -> bool:
    src = results / "sim_efficiency.json"
    if not src.exists():
        return False
    rows = json.loads(src.read_text())["rows"]
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, c in enumerate(conds):
        rs = sorted([r for r in rows if r["condition"] == c], key=lambda r: r["corpus_size"])
        ax.errorbar([r["corpus_size"] for r in rs], [r["acc_mean"] for r in rs],
                    yerr=[r["acc_std"] for r in rs], marker="o", ms=4, capsize=3,
                    label=c, color=PALETTE[i % len(PALETTE)])
    ax.set_xscale("log")
    ax.set_xlabel("corpus size (episodes)")
    ax.set_ylabel("lexicon accuracy")
    ax.set_title("Contrastive setting: no reliable benefit (scope condition)",
                 loc="left", fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "efficiency.png", dpi=180)
    plt.close(fig)
    return True


def fig_training(runs: Path, out: Path) -> bool:
    hists = sorted(runs.glob("*/history.json")) if runs.exists() else []
    if not hists:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, h in enumerate(hists):
        rows = json.loads(h.read_text())
        pts = [(r["step"], r["pv_accuracy"]) for r in rows if "pv_accuracy" in r]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3,
                    label=h.parent.name, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("training step")
    ax.set_ylabel("picture-vocabulary accuracy")
    ax.set_title("Training runs", loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "training.png", dpi=180)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    made = {
        "ar_ablation": fig_ar_ablation(args.results, args.out),
        "text_blind_audit": fig_text_blind(args.results, args.out),
        "efficiency": fig_efficiency(args.results, args.out),
        "training": fig_training(args.runs, args.out),
    }
    for name, ok in made.items():
        print(f"  {'wrote' if ok else 'skipped (no input)'}: {name}")


if __name__ == "__main__":
    main()
