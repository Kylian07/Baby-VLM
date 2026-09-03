# GAVAGAI

**Referential alignment for infant-scale vision–language models**, targeting the
[BabyVLM Workshop @ NeurIPS 2026](https://babyvlm.github.io/#cfp) (deadline 8 Sep 2026, ≤8 pages, non-archival).

Named for Quine's *gavagai*: a caregiver's utterance and the scene in front of an
infant are a bag of words and a bag of things with unknown correspondence.

---

## The starting observation

BabyVLM-V2's from-scratch 1.1B "baby model" shows a sharp dissociation on **DevCV Toolbox**:

| near-ceiling | | near-chance | |
|---|---|---|---|
| Left/Right | 96.4 | Picture Vocabulary | 32.4 (chance 25) |
| Spatial Details | 92.8 | Localization | 37.8 (chance 25) |
| Who-Has-More (synth) | 99.7 | VDR-binary | 54.6 (chance 50) |

It looks like a model with excellent perception and no word knowledge. **Part of that
turns out to be an artefact of what the tasks measure.**

## Finding 1 — some of these tasks are solvable without reading the question

`scripts/text_blind_audit.py` answers each multiple-choice item using a **32×32
luminance + RGB-histogram** descriptor: no training, no network, no language.

Pooled over the public SAYCam / BabyView / Ego4D samples:

| task | text-blind image-match | baby model | human | chance |
|---|---|---|---|---|
| **Left/Right** | **1.00** (24/24, 95% CI [0.86, 1.00]) | 96.4 | 94.5 | 33.3 |
| NIH Spatial | **1.00** (10/10, CI [0.72, 1.00]) | 92.8 | 100 | 33.3 |
| spatialdetails (SAYCam) | 0.33 (CI [0.10, 0.70]) | — | — | 33.3 |
| Picture Vocabulary | n/a — no query image to match | 32.4 | 91.8 | 25.0 |

Left/Right, where the baby model scores **above the human ceiling**, is fully solved by a
matcher that never reads the prompt. Picture Vocabulary is not — a useful control showing
the baseline is not simply always winning.

> **Caveat, stated up front.** `n` is 6–24 per task on the bundled website samples, hence
> the Wilson intervals. These numbers must be regenerated on the full public Ego4D release
> before being quoted. The script does that in one command.

## Finding 2 — the residual gap is real, and it is about referential binding

The tasks that survive the audit (Picture Vocabulary, Localization) are exactly those
requiring a word→referent mapping. Every training stage of the baby model is **LLaVA-style
next-token prediction**: there is no contrastive term, no batch-level marginal correction,
and no latent word↔region variable. Utterance perplexity can be driven down from global
gist plus language priors without ever binding a word to a thing.

## Proposed fix — cross-situational grounding as balanced optimal transport

A cheap auxiliary loss adding the missing latent variable. Per episode, solve a
semi-relaxed entropic OT problem between content words and object slots, with

* a **null bin**, so a word may refer to nothing visible (most caregiver tokens do not); and
* a **column marginal constraint** = mutual exclusivity, which makes hub collapse *infeasible*
  rather than merely disfavoured.

Two knobs generate the whole family — `ρ = 0` provably reduces to the row-wise softmax of
standard region–word contrastive learning, so every ablation changes one scalar rather than
swapping code paths. See [`METHOD.md`](METHOD.md) for the derivation and propositions.

### A scope condition we established by trying to break it

In controlled simulation the exclusivity constraint gives **no benefit** when a strong
batch-level InfoNCE term is already present: that term's negatives come from the marginal
and already suppress hub collapse. This is reported, not hidden — it is why the method is
aimed at *autoregressively* trained baby VLMs, which have no such term.

---

## Layout

```
gavagai/
  ot.py         log-domain semi-relaxed Sinkhorn with a null bin
  losses.py     E-step / M-step objective + hub-collapse diagnostic
  lexicon.py    persistent word x concept lexicon with PMI feedback
  sim.py        controlled cross-situational world; contrastive + AR learners
  models.py     16.8M-param ViT-S/16 grounding model (T4-sized, from scratch)
  data/devcv.py DevCV-Toolbox loader + zero-shot PV / localization evaluators
scripts/
  text_blind_audit.py   benchmark audit (Finding 1)
  run_simulation.py     controlled experiments
tests/          executable checks for every proposition in METHOD.md
```

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest tests/ -q

# Finding 1, on the full public Ego4D release
hf download wsashawn/devcv_toolbox_ego4d --repo-type dataset --local-dir data/Ego4D
python scripts/text_blind_audit.py --roots data/Ego4D

# Controlled experiments (CPU, minutes)
python scripts/run_simulation.py efficiency
python scripts/run_simulation.py rho
```

## Data

SAYCam and BabyView are Databrary-gated. Every number here is reproducible from the
**public Ego4D variants** released by the BabyVLM-V2 authors:
`wsashawn/devcv_toolbox_ego4d`, `wsashawn/babyllava_v2_instruction_ft_Ego4D`, and the
phase-0/2/3 checkpoints. A SAYCam adapter is included but is not on the critical path.
