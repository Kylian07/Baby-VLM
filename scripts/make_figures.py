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
    """What each DevCV task can be answered by, without reading the prompt.

    Every task appears, including those no baseline can touch: those are the
    controls, and dropping them would make the baselines look like they win
    everywhere. Palette is categorical slots 1-2 of the reference palette
    (validated: adjacent CVD dE 24.7, normal-vision 33.6, contrast >= 3:1).
    """
    src = results / "text_blind_audit.json"
    if not src.exists():
        return False
    rows = json.loads(src.read_text())

    BLUE, ORANGE = "#2a78d6", "#eb6834"
    MUTED, INK = "#8a8880", "#0b0b0b"

    recs = []
    for t, r in rows.items():
        if "dup_acc" in r:
            recs.append((t, r["dup_acc"], r["dup_ci"], r["dup_n"], ORANGE,
                         "duplicate file", r.get("degenerate")))
        elif "match_acc" in r:
            recs.append((t, r["match_acc"], r["match_ci"], r["match_n"], BLUE,
                         "image match", r.get("degenerate")))
        else:
            recs.append((t, None, None, r["n_items"], None, None, None))
    # Solvable first (the finding), controls last.
    recs.sort(key=lambda x: (0 if x[1] is not None else 1, -(x[1] or 0)))

    fig, ax = plt.subplots(figsize=(9.0, 0.5 * len(recs) + 2.0))
    labels = []
    for i, (task, acc, ci, n, color, kind, degen) in enumerate(recs):
        y = len(recs) - 1 - i
        chance = rows[task]["chance"]
        if acc is None:
            labels.append(f"{task}\n(n={n})")
            ax.text(0.012, y, "no text-blind baseline applies",
                    va="center", fontsize=8.5, color=MUTED, style="italic")
            continue

        hatch = "///" if degen else None
        ax.barh(y, acc, height=0.52, color=color, zorder=2,
                hatch=hatch, edgecolor="white" if degen else "none", linewidth=0)
        ax.errorbar(acc, y, xerr=[[acc - ci[0]], [ci[1] - acc]], fmt="none",
                    ecolor=INK, elinewidth=1.1, capsize=3, zorder=3)
        # Long bars label inside so nothing is clipped at the axis edge.
        if acc > 0.85:
            ax.text(acc - 0.015, y, f"{acc:.2f}", va="center", ha="right",
                    fontsize=9.5, color="white", fontweight="bold", zorder=4)
        else:
            ax.text(acc + 0.03, y, f"{acc:.2f}", va="center", ha="left",
                    fontsize=9.5, color=INK, fontweight="bold", zorder=4)
        if degen:
            # Kept off the y-axis label, which would widen the left margin and
            # squeeze the plot.
            ax.text(acc + 0.10, y, "all options identical to the query - no signal",
                    va="center", ha="left", fontsize=8.5, color=MUTED, style="italic")
        labels.append(f"{task}\n(n={n})")

    for i, (task, *_rest) in enumerate(recs):
        y = len(recs) - 1 - i
        c = rows[task]["chance"]
        ax.plot([c, c], [y - 0.3, y + 0.3], color=INK, lw=1.6, zorder=5)

    ax.set_yticks(list(range(len(recs))))
    ax.set_yticklabels(list(reversed(labels)), fontsize=9)
    ax.set_xlim(0, 1.09)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("accuracy of a baseline that never reads the prompt\n"
                  "vertical rule = chance", fontsize=9)
    ax.set_title("DevCV-Toolbox: what can be answered without perception",
                 loc="left", fontweight="bold", fontsize=12, pad=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="0.9", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=ORANGE, label="duplicate-file baseline (filenames only)"),
        Patch(facecolor=BLUE, label="image-match baseline (32x32 histogram)"),
        Patch(facecolor=BLUE, hatch="///", edgecolor="white",
              label="no signal in the released data"),
    ], fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, -0.02))
    fig.tight_layout()
    fig.savefig(out / "text_blind_audit.png", dpi=180, bbox_inches="tight")
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
    ax.set_title("Contrastive setting: accuracy vs. corpus size",
                 loc="left", fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "efficiency.png", dpi=180)
    plt.close(fig)
    return True


def fig_rho(results: Path, out: Path) -> bool:
    """Accuracy and hub rate against the mutual-exclusivity strength.

    The two curves together are the argument: hub rate falls monotonically as
    the column constraint tightens (Proposition 2), while accuracy peaks at an
    interior optimum because full exclusivity over-constrains.
    """
    src = results / "sim_rho.json"
    if not src.exists():
        return False
    rows = json.loads(src.read_text())["rows"]

    def key(r):
        v = r["condition"].split("=", 1)[1]
        return float("inf") if v == "None" else float(v)

    rows = sorted(rows, key=key)
    xs = [key(r) for r in rows]
    xpos = list(range(len(rows)))
    labels = ["0" if x == 0 else ("∞" if x == float("inf") else f"{x:g}") for x in xs]

    fig, ax = plt.subplots(figsize=(8.2, 4))
    ax.errorbar(xpos, [r["acc_mean"] for r in rows], yerr=[r["acc_std"] for r in rows],
                marker="o", ms=5, capsize=3, color=PALETTE[0], label="lexicon accuracy")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("mutual-exclusivity strength  ρ"
                  "\n(ρ=0 is exactly region–word contrastive learning)")
    ax.set_ylabel("lexicon accuracy", color=PALETTE[0])
    ax.spines[["top"]].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(xpos, [r["hub_mean"] for r in rows], marker="s", ms=4, ls="--",
             color=PALETTE[1], label="hub rate")
    ax2.set_ylabel("hub rate (words → an unnamed background object)", color=PALETTE[1], fontsize=9)
    ax2.spines[["top"]].set_visible(False)
    ax2.set_ylim(0, max(r["hub_mean"] for r in rows) * 1.6 + 1e-3)

    best = max(rows, key=lambda r: r["acc_mean"])
    ax.annotate("interior optimum", xy=(xpos[rows.index(best)], best["acc_mean"]),
                xytext=(10, -28), textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    ax.set_title("Exclusivity strength: accuracy peaks, hubs trend down",
                 loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "rho_sweep.png", dpi=180)
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
        "rho_sweep": fig_rho(args.results, args.out),
        "training": fig_training(args.runs, args.out),
    }
    for name, ok in made.items():
        print(f"  {'wrote' if ok else 'skipped (no input)'}: {name}")


if __name__ == "__main__":
    main()
