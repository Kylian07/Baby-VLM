#!/usr/bin/env python3
"""Build notebooks/BabyGOT_T4.ipynb from a small spec (no nbformat dependency)."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "notebooks", "BabyGOT_T4.ipynb")


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


cells = [
    md("""# BabyGOT — Grounded Optimal-Transport Pretraining (Kaggle T4)

**BabyVLM Workshop (NeurIPS 2026)** — *"Toward Developmentally Plausible Multimodal Systems."*

This notebook trains a small vision-language model **from scratch on a single T4 GPU**
with a novel, mathematically-grounded objective:

- **Referential optimal transport (Sinkhorn)** aligns *patch* tokens with *word* tokens
  through a soft, doubly-stochastic coupling $P^\\star$ — a generalisation of the global
  CLIP/CVCL (InfoNCE) and FILIP alignment losses.
- **Grounded summary tokens** preserve *where / how many*, fixing the "global-embedding
  bottleneck" of the BabyVLM / BabyVLM-V2 / LLaVA connector.
- A **token-wise gate** learns the content-word / function-word split with no supervision.

Full mathematics: `paper/paper.md`.  Read the recommended papers first:
BabyVLM (2504.09426), BabyVLM-V2 (2512.10932), Looking to Learn (2025.babylm-main.15),
LLaVA (2304.08485), BabyView (2406.10447), SAYCam (Open Mind 2022)."""),

    md("""### 1. Environment (T4 = 16 GB VRAM)

The code needs only `torch` + `numpy`. On Kaggle both are already installed.
If you uploaded this repo as a dataset / used "Add-Only", adjust the paths below."""),

    code("""import sys, os, subprocess

# Locate the repo: either we are already inside it, or we clone it.
if os.path.exists('src/babygot'):
    REPO = '.'
elif os.path.exists('Baby-VLM/src/babygot'):
    REPO = 'Baby-VLM'
else:
    subprocess.run(['git', 'clone', 'https://github.com/Kylian07/Baby-VLM.git'],
                   check=True)
    REPO = 'Baby-VLM'
sys.path.insert(0, os.path.join(REPO, 'src'))
os.chdir(REPO)

import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),
      '| device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"""),

    md("""### 2. Smoke test (tiny model, ~30 s)

Confirms the pipeline end-to-end before committing GPU time."""),

    code("""!PYTHONPATH=src python -m babygot.run --method babygot --tiny"""),

    md("""### 3. Full run on the T4 (base config, ~6M params)

Pretraining (from scratch) + instruction tuning + the 10-probe DevCV-lite suite.
The model trains in minutes on a T4."""),

    code("""!PYTHONPATH=src python -m babygot.run --method babygot \\
    --steps 4000 --n-train 4000 --n-eval 200 --save-dir runs"""),

    md("""### 4. The paper's ablation table (all methods, same budget & seed)

- `babygot` — ours
- `global_clip` — CLIP/CVCL-style global InfoNCE + generative LM
- `babyllava` — BabyVLM-style single global token + autoregression
- `no_gate`, `no_ot` — component ablations"""),

    code("""!PYTHONPATH=src python -m babygot.run --all --small"""),

    md("""### 5. Results + interpretability

Every run writes `runs/<method>.json` with per-probe accuracy plus two
interpretability metrics produced by the model itself:

- `analysis.mean_localization_error_patch` — distance between the OT coupling's
  centre of mass (the model's "pointing") and the true object, in patch units;
- `analysis.gate_content` / `analysis.gate_function` — the learned
  content/function-word gate."""),

    code("""import json, glob
import matplotlib.pyplot as plt

rows = {}
for f in sorted(glob.glob('runs/*.json')):
    rows[f.split('/')[-1][:-5]] = json.load(open(f))

probes = [k for k in next(iter(rows.values())) if k != 'overall_choice_acc' and k != 'analysis']
fig, ax = plt.subplots(figsize=(12, 5))
for name, r in rows.items():
    vals = [r[p].get('acc', r[p].get('f1', 0.0)) for p in probes]
    ax.plot(probes, vals, marker='o', label=name)
ax.axhline(0.25, ls='--', c='grey', lw=1)
ax.set_ylim(0, 1); ax.legend(); ax.set_ylabel('accuracy / F1')
plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.show()

for name, r in rows.items():
    a = r.get('analysis', {})
    if a:
        print(f"{name:12s}  loc_err={a.get('mean_localization_error_patch', float('nan')):.2f}"
              f"  gate_content={a.get('gate_content', 0):.3f}  gate_function={a.get('gate_function', 0):.3f}")"""),

    md("""### 6. What to expect

- **Picture vocabulary** saturates quickly for *all* methods (naming is easy).
- **Localization, counting, spatial details, memory** separate the methods:
  BabyGOT's transport-aligned, spatially-preserving representations retain the
  location/count information that global pooling destroys.
- The **gate** opens for content words and closes for function words without
  supervision; the **OT coupling** gives a word→region pointing map.

For a deep dive, read `paper/paper.md` and `src/babygot/transport.py`."""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", os.path.abspath(OUT))
