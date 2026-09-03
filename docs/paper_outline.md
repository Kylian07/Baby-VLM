# 8-page workshop paper — section plan

Target: BabyVLM Workshop @ NeurIPS 2026, deadline 8 Sep 2026, non-archival, double-blind.

The ordering below is deliberate: the contribution that is a **measurement** comes first, and
the contribution that is a **bet** comes second. If the Ego4D training run does not work out,
the paper still stands on §3 alone.

---

## Title (candidates)

- *What Does a Baby VLM Actually Know? Text-Blind Baselines and Referential Alignment for Infant-Scale Vision–Language Models*
- *Grounding Without Reading the Question: Auditing DevCV Toolbox and Fixing Referential Alignment*

## §1 Introduction (~0.75 p)

The dissociation in BabyVLM-V2: 96.4 on Left/Right, 32.4 on Picture Vocabulary against a
chance floor of 25. Two questions follow — is the dissociation real, and if partly real,
what causes it? Contributions listed in decreasing order of certainty.

## §2 Background (~0.75 p)

SAYCam and its Databrary gate; BabyVLM-V1/V2 and DevCV Toolbox; that **every BabyVLM-V2
training stage is next-token prediction**. Cite the Third BabyLM Challenge finding that no
submission has beaten the multimodal baselines in two consecutive rounds — the multimodal
track is where the field is stuck.

## §3 A text-blind audit of DevCV Toolbox (~1.5 p) — *the measurement*

Method: a 32×32 luminance + RGB-histogram matcher, no training, no access to the prompt;
plus a most-frequent-answer baseline; Wilson intervals throughout.

Result: Left/Right **1.00 (24/24, CI [0.86, 1.00])** where the baby model scores 96.4 and
humans 94.5; NIH Spatial 1.00 (10/10). Picture Vocabulary is *not* solvable this way — the
control that the baseline is not simply always winning.

Reading: the high scores are not evidence of grounding, and the grounding-diagnostic subset
of DevCV is smaller than it appears. Propose reporting a text-blind row alongside every
future DevCV table. **Regenerate on the full public Ego4D release before submitting.**

Figure: `docs/figures/text_blind_audit.png`.

## §4 Diagnosis: no referential binding pressure (~0.75 p)

An autoregressive captioning objective has no latent word↔region variable. Utterance
perplexity is reducible from global gist plus language priors. Note the corroboration from
*Looking to Learn*: their learned gate favours visual cues for content words and linguistic
cues for function words, i.e. models will discover the referential/non-referential split if
given the capacity — we make it structural.

## §5 Method: cross-situational grounding as balanced OT (~1.5 p)

E-step (semi-relaxed entropic OT with a null bin), cross-situational lexicon with PMI
feedback, M-step (referentially weighted soft-target InfoNCE). Propositions 1–3, with
Proposition 1 (`ρ=0` *is* region–word contrastive learning) doing the heavy lifting: it makes
every ablation a one-scalar change. State Proposition 4 (Sinkhorn ≠ PMI ranking) as a caveat —
it costs nothing and signals care.

Figure: one schematic of the coupling matrix with the null column.

## §6 Experiments (~2 p)

1. **Captioning ± alignment** across ambiguity regimes (`docs/figures/ar_ablation.png`).
   Headline: 0.392 → 0.700 clean, 0.242 → 0.400 moderate, and naive alignment
   **collapsing to 0.000** under realistic ambiguity.
2. **ρ sweep** (`docs/figures/rho_sweep.png`): interior optimum, hub rate down.
3. **Psychometric fit**: limited encoding, 2 objects/trial, RMSE 0.066 against Yu & Smith.
4. **Ego4D run** — the one still to do. Zero-shot Picture Vocabulary and Localization from
   `scripts/evaluate.py`.

## §7 Limitations (~0.5 p)

Verbatim from `METHOD.md` §7, including the retraction in §6.2b and the floor-limited
realistic regime. A workshop reviewer rewards this; a hidden version of it gets found.

## §8 Conclusion (~0.25 p)

---

## Priority if time runs short

1. §3 audit on the full Ego4D release — highest value per hour, and it cannot fail.
2. §6.1–6.3 are already measured and only need writing up.
3. §6.4 Ego4D training is the only item that can come back negative. If it does, report it
   and narrow the paper to the audit plus the controlled study; the abstract should be
   written so that outcome does not require restructuring.

## Anonymisation checklist

Double-blind: strip the GitHub URL from the camera-ready-facing version, or replace with an
anonymised mirror. Check `README.md`, `METHOD.md`, and the notebook's clone cell.
