# BabyGOT — Grounded Optimal-Transport Pretraining for Baby-scale VLMs

> A novel, mathematically-grounded method + full codebase for the **BabyVLM
> Workshop (NeurIPS 2026)**: *"Toward Developmentally Plausible Multimodal
> Systems."*  Trains a small vision-language model **from scratch on a single
> Kaggle T4 GPU**, with the complete training/evaluation code, a proof that the
> pooled BabyVLM/LLaVA connectors have *zero* localization capacity, mechanism
> ablations, and honest reporting.

---

## The idea in one paragraph

The BabyVLM / BabyVLM-V2 / LLaVA recipe squeezes an entire image into **one
global token** before feeding it to the language model. That destroys *where*
and *how many* — which is exactly why baby models score 37.8 vs 98.2 (humans) on
Localization and 32.4 vs 100 on Spatial Details in the DevCV Toolbox, while
matching large models on object naming. **BabyGOT** models the word↔image-region
correspondence as a **latent variable** and infers it with **entropy-regularised
optimal transport** (Sinkhorn): patch tokens and word tokens are aligned through
a soft, doubly-stochastic *referential coupling* $P^\star$. That coupling
(i) is the training signal (a principled generalisation of InfoNCE/CLIP and
FILIP), (ii) feeds a compact set of **spatially-ordered summary tokens** to the
LM so location and count survive, and (iii) drives a **token-wise gate** that
*can* learn the content-word/function-word split with no supervision (an honest
negative result at our small demonstration budget is reported in the paper).
The whole objective is a **variational EM lower bound** on
$\log p(\text{caption}\mid\text{image})$.

Full mathematics: [`paper/paper.md`](paper/paper.md).  Key claims and how they
relate to the recommended reading:

| idea in BabyGOT | addresses / builds on |
|---|---|
| referential OT coupling | BabyVLM-V1's call for hybrid generative+contrastive models; the "global-embedding bottleneck" identified by *Looking to Learn* (Ganescu et al. 2025) |
| spatially-ordered summary tokens (not one pooled vector) | LLaVA / BabyLLaVA / BabyVLM-V2 connector; DevCV failures on Localization / Spatial Details / Counting |
| token-wise dynamic gate | *Looking to Learn* (BabyLM 2025) |
| four-stage from-scratch recipe | BabyVLM-V2 training paradigm; LLaVA feature-alignment→joint→instruction |
| cross-situational referential learning | SAYCam/BabyView line of work; Smith & Yu (2007); MEWL |

## Quick start

```bash
# deps: torch + numpy only
pip install -r requirements.txt

# (recommended) install the package so `python -m babygot.run` works from anywhere
pip install -e .

# 30-second smoke test (CPU, tiny model)
python -m babygot.run --method babygot --tiny

# fast but meaningful run (~1.9M params)
python -m babygot.run --method babygot --small

# full run on a Kaggle T4 GPU
python -m babygot.run --method babygot --steps 4000 --n-train 4000 --n-eval 200

# the paper's ablation table (all 5 methods, same budget & seed)
python -m babygot.run --all --small
```

> If you prefer not to install, run from the repo root with the package on the
> path: `PYTHONPATH=src python -m babygot.run ...`

Methods: `babygot` (ours), `global_clip` (CLIP/CVCL-style), `babyllava`
(BabyVLM-style global token + AR), `no_gate`, `no_ot` (FILIP-style surrogate).

## Repository layout

```
paper/paper.md          # the paper: full maths, related work, experiments
paper/references.bib    # BibTeX
src/babygot/
  config.py             # all hyperparameters; tiny/small/base presets
  vision.py             # from-scratch ViT → patch features (keeps "where")
  text.py               # from-scratch tiny GPT (generative arm)
  transport.py          # entropy-regularised OT / Sinkhorn (the core)
  fusion.py             # spatially-ordered summary tokens + token-wise gate
  model.py              # BabyGOT (combines the above; baselines included)
  data.py               # synthetic "infant-like" scene generator + instruction data
  benchmarks.py         # DevCV-style probe suite (10 tasks)
  evaluate.py           # maximum-likelihood scoring (fair across methods)
  analysis.py           # word→region pointing maps + gate interpretability
  train.py              # AMP, grad-accum, warmup, SFT stage — T4 friendly
  run.py                # CLI + ablation runner
notebooks/BabyGOT_T4.ipynb   # drop-in Kaggle notebook
```

## What the benchmark looks like

Because SAYCam/BabyView are IRB-gated, the code ships a deterministic synthetic
generator of **coloured geometric scenes** (the same stimulus family as the NIH
Baby Toolbox / DevCV).  Each scene = image + child-directed utterance +
structured annotation.  Probes (constructed from ground truth, never from model
output): picture vocabulary, counting, localization (H/V), who-has-more, spatial
details, visual delayed response (memory), a two-word compositional test, and
captioning — a "DevCV-lite".

Plugging in real data is a one-item contract: `{"image": (3,H,W) float32 in
[0,1], "caption": str}` — see `src/babygot/data.py::FromDiskDataset`.

## Interpretability for free

After any run, `runs/<method>.json` includes:

- `mean_localization_error_patch` — how far the OT coupling's centre of mass
  (the model's "pointing") is from the true object, in patch units;
- `gate_content` vs `gate_function` — the learned content/function-word split.

## Notes for reviewers / organisers

- Submission target: **BabyVLM Workshop 2026** (NeurIPS, Atlanta). Paper ≤ 8
  pages, non-archival; deadline **Sep 8, 2026**.
- The code is deliberately dependency-light (PyTorch + NumPy) so the mathematics
  is auditable and the whole thing runs anywhere, including a single T4.
- CPU results in `runs/` are for plumbing validation; the headline numbers come
  from the T4 command above (base config, ~6M params).
