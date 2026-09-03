# Reading notes — the BabyVLM workshop's recommended papers

What each listed paper establishes, and how this project relates to it. Where a
paper could not be retrieved in full, that is stated rather than glossed over.

---

## Training data

### SAYCam (Sullivan, Mei, Perfors, Wojcik & Frank, *Open Mind*, 2022)
Head-mounted camera recordings from three children, roughly two hours per week
from 6 to 32 months; ~478 hours total. The reference corpus for "what a child
actually sees and hears". **Databrary-gated**, which is the single biggest
practical constraint on this line of work: results on it are not directly
reproducible by a reviewer without an approved account.

### The BabyView Dataset (Long et al., 2024)
A second, higher-resolution longitudinal egocentric corpus of infants' and young
children's everyday experience. Used in DevCV Toolbox as an out-of-distribution
evaluation source alongside SAYCam. Also gated.

**Consequence for this project.** Everything here is built to run on the
**public Ego4D variants** the BabyVLM-V2 authors released, with a SAYCam adapter
present but off the critical path.

---

## Baselines

### BabyVLM (Wang et al., ICCV 2025) — "V1"
Three contributions: in-domain evaluation tasks derived from SAYCam; synthetic
data augmentation that distils CC3M into simplified, child-directed form with
GPT-4o; and **BabyLLaVA**, a generative VLM trained entirely on developmentally
plausible data. The authors' own later assessment is that V1 used about a third
of SAYCam, had no instruction tuning, had benchmarks not grounded in any
psychology instrument, and produced models with *near-zero open-set performance*
requiring logit post-processing to evaluate.

### BabyVLM-V2 (Wang et al., CVPR 2026)
The paper this project engages with most directly.

*Data.* Full SAYCam (478 h): ~181k video–utterance pairs, ~768k image–utterance
pairs, ~63k interleaved multi-turn sequences, plus a ~150k instruction set.

*Model.* A 1.1B "baby model" trained from scratch in four stages — TinyLLaMA-1.1B
on transcribed SAYCam (283k), a DINO vision backbone on 1085k frames (Orhan et
al.'s pipeline), MLP-connector alignment, joint pretraining, then instruction
tuning. **Every stage is next-token prediction; there is no contrastive term
anywhere.** This is the observation the present work is built on.

*Benchmark.* DevCV Toolbox — all vision-related measures of the NIH Baby
Toolbox® adapted into ten multimodal tasks.

*The result we take as our starting point.* The baby model reaches 96.4 on
Left/Right and 92.8 on Spatial Details, but 32.4 on Picture Vocabulary against a
chance floor of 25.0 and a human ceiling of 91.8, and 37.8 on Localization
against a chance floor of 25.0.

### Visual Instruction Tuning / LLaVA (Liu, Li, Wu & Lee, NeurIPS 2023)
The architecture template: frozen vision encoder, MLP connector, LM, trained by
autoregressive prediction of the response. BabyVLM-V2's baby model is a
from-scratch LLaVA. Relevant here because the objective supplies **no latent
word↔region variable** — nothing in the loss requires a particular word to be
explained by a particular region.

### Looking to Learn (Ganescu, Salhan, Caines & Buttery, BabyLM 2025)
A lightweight decoder with token-wise dynamic gating to fuse linguistic and
visual cues, feature modulation and channel attention, plus auxiliary contrastive
objectives for visual grounding. Competitive on BLiMP, BLiMP-Supplement, EWoK,
Winoground and VQA.

**The finding that matters most for us:** the learned gate, with no explicit
supervision, *favours visual cues for content words and linguistic cues for
function words*.

This is independent evidence for the premise behind our null bin. They show a
model will discover the referential/non-referential split if given the capacity
to; we make that split an explicit, structured constraint — a word may be
assigned to the null bin instead of being forced onto a region — and derive it
from an unbalanced transport problem rather than a learned scalar gate. Two of
this paper's authors organise the BabyVLM workshop.

---

## Developmental grounding

### NIH Baby Toolbox
Validated measures of cognitive, sensory, motor and social-emotional development
for ages 0–3, released February 2025. DevCV Toolbox adapts its vision-related
measures with deliberately minimal changes to preserve developmental fidelity.

Our audit does not question that fidelity. It asks a narrower and separate
question: once a *touch-and-point* instrument for infants is rendered as
multiple-choice items over images, how many of those items can be answered
without reading the question at all? For Left/Right, the answer appears to be
all of them.

---

## Findings

### Findings of the Third BabyLM Challenge (Charpentier et al., BabyLM 2025)
Multimodal models were reintroduced with GIT and Flamingo baselines. **No
submission outperformed the multimodal baselines** — and none did in the second
challenge either, making it two consecutive rounds. Meanwhile the strict-small
and interaction tracks *did* see submissions beat their baselines, and the report
notes that new **training objectives and architectures** tend to produce the
best-performing approaches.

Two things follow. First, the multimodal track is where the field is most stuck,
so it is the right place to work. Second, the mechanism most likely to move it is
an objective-level change rather than a scaling or architecture tweak — which is
what this project proposes.

---

## Beyond the reading list

### CVCL — Vong, Wang, Orhan & Lake, *Science* 2024
Contrastive training on frames paired with transcribed child-directed speech
from a single child in SAYCam; acquires word-referent mappings that generalise to
novel referents. Whole-frame ↔ whole-utterance, with **no latent word↔region
variable**, which is the gap our objective fills. Their reported statistic that
the named referent is visible for only a minority of utterances is the empirical
basis for our null bin and for the `absent_ref_prob` parameter in the simulator.

### Yu & Smith, *Psychological Science* 2007
Adults learn 18 word-referent pairs from 27 trials of ambiguous co-occurrence:
0.889 (2×2), 0.778 (3×3), 0.556 (4×4). The psychometric target. An unconstrained
ideal observer is at ceiling on this design, so the informative question is what
capacity limit reproduces the human ordering.

### Frank, Goodman & Tenenbaum, 2009
Bayesian cross-situational word learning with an explicit lexicon variable and a
non-referential state. Our E-step is a differentiable, GPU-scale, online-EM
version of the same generative story, with mutual exclusivity expressed as a
transport constraint rather than a prior over lexicons. (Michael C. Frank is an
invited speaker and on the challenge steering committee.)

### OTTER (Wu et al., 2021)
Optimal transport to soften noisy image–text labels — the closest prior art.
Within-batch and caption-level. Ours is within-episode at word↔slot granularity,
carries a null bin, and maintains a persistent cross-situational lexicon across
episodes.

---

## Retrieval note

`arxiv.org`, `aclanthology.org`, `huggingface.co` and `openaccess.thecvf.com`
were unreachable from the environment these notes were written in. BabyVLM-V2 was
reconstructed from its project page, published results table and recipe figure,
and the DevCV-Toolbox / BabyLLaVA-V2 source repositories; the remainder from
search summaries. Numbers quoted from BabyVLM-V2's results table were read
directly from the published figure and should be checked against the PDF before
being cited in a submission.
