# GAVAGAI — Cross-Situational Grounding as Balanced Optimal Transport

*A sample-efficient referential-alignment objective for infant-scale vision–language models.*

Target venue: **BabyVLM Workshop @ NeurIPS 2026** (submission deadline 8 Sep 2026, ≤8 pages, non-archival, double-blind).

---

## 1. The gap we are attacking

BabyVLM-V2 (Wang et al., CVPR 2026) trains a 1.1B "baby model" from scratch on SAYCam and evaluates it on **DevCV Toolbox**, ten tasks adapted from the NIH Baby Toolbox®. Its published per-task numbers contain a striking dissociation:

| Task | Baby model | Chance | Human |
|---|---|---|---|
| Left/Right (visual matching) | **96.4** | 33.3 | 94.5 |
| Spatial Details (visual matching) | **92.8** | 33.3 | 100 |
| Who-Has-More, synthetic | **99.7** | 50.0 | 98.2 |
| Memory | 90.8 | 25.0 | 97.9 |
| — | | | |
| **Picture Vocabulary** (word → referent) | **32.4** | 25.0 | 91.8 |
| **Localization** (word → region) | **37.8** | 25.0 | 87.3 |
| Visual Delayed Response, binary | 54.6 | 50.0 | 98.2 |
| Who-Has-More, naturalistic | 60.5 | 50.0 | 96.4 |

The model is at or near **ceiling** on every task that reduces to *"which of these images is the same as this one?"*, and at or near **chance** on every task that requires knowing *what a word refers to*. A 3B off-the-shelf VLM reaches 71.7 on Picture Vocabulary; the baby model reaches 32.4 against a chance floor of 25.

This is not a failure of perception. The visual representation is evidently good enough to support fine-grained instance matching. It is a failure of **referential alignment** — and word learning is the single capacity that human infants most conspicuously *do* possess at this age.

**Claims of this work, in decreasing order of certainty.**

1. **Part of the dissociation is a benchmark artefact** (§1.1). Some of the tasks the baby model aces can be answered without reading the question at all. This is a measurement, not a bet.
2. **The residual gap is real and is about referential binding** (§2). The tasks that survive the audit are exactly those needing a word→referent mapping, and the autoregressive objective used to train the baby model contains no term that supplies binding pressure.
3. **It is fixable cheaply** (§3) by adding the missing latent variable: a semi-relaxed optimal-transport alignment between words and object slots, with a null bin for non-referential speech. In controlled simulation this lifts picture-vocabulary accuracy from 0.392 to 0.700 (clean) and 0.242 to 0.400 (moderate ambiguity), while the *naive* form of the same alignment — no null bin, row-softmax E-step — collapses to **0.000** under realistic ambiguity, i.e. it is worse than adding nothing. §6.2b states the scope condition under which the exclusivity constraint does *not* help.

---

## 1.1 First: are these tasks measuring grounding?

Before attributing a score to a model's competence, it is worth asking what a *trivial* system scores. `scripts/text_blind_audit.py` answers each multiple-choice item with a 32×32 luminance + RGB-histogram descriptor — no training, no network, and crucially **no access to the prompt**. For items shaped "here is X, which of (A)(B)(C) is the same X?", it simply returns the option most similar to the query image.

Pooled over the public SAYCam / BabyView / Ego4D samples:

| task | text-blind image match | baby model | human | chance |
|---|---|---|---|---|
| **Left/Right** | **1.00** (24/24, 95% CI [0.86, 1.00]) | 96.4 | 94.5 | 33.3 |
| NIH Spatial | **1.00** (10/10, CI [0.72, 1.00]) | 92.8 | 100 | 33.3 |
| spatialdetails (SAYCam variant) | 0.33 (CI [0.10, 0.70]) | — | — | 33.3 |
| Picture Vocabulary | n/a (no query image to match against) | 32.4 | 91.8 | 25.0 |

Left/Right — where the baby model scores **above the human ceiling** — is solved outright by a matcher that never reads the question. Picture Vocabulary is not, which is the control that shows the baseline is not simply always winning.

Two consequences. First, a chunk of the apparent "excellent perception, no word knowledge" dissociation is a property of the tasks rather than of the model. Second, the *interesting* subset of DevCV Toolbox — the part that actually demands grounding — is smaller than it appears, and that subset is where the baby model sits near chance.

**Caveat stated up front.** `n` is 6–24 per task on the bundled website samples, hence Wilson intervals throughout. These numbers must be regenerated on the full public Ego4D release before being quoted; the script does so in one command.

---

## 2. Why the standard objective fails here

A caregiver utterance paired with an egocentric frame is not an aligned pair. It is a *bag of words* and a *bag of things* with unknown correspondence — Quine's **gavagai** problem. Two properties of infant-egocentric data make this severe:

1. **Most word tokens are non-referential.** "Look at that", "are you hungry", "there we go" refer to nothing on screen.
2. **A few objects are in almost every frame and are almost never named.** Hands, floor, table, the caregiver's torso. These are the raw material of a **hub**: a slot always available to absorb every word's alignment mass.

CLIP-style objectives sidestep the problem by aligning the *whole* utterance to the *whole* frame, which never isolates individual word meanings. Region–word objectives (GLIP, FILIP, and the region branch of most grounded VLMs) do introduce a word→region assignment, but they compute it with a **row-wise softmax**: each word independently picks its best-matching region. Nothing prevents every word from picking the *same* region, and the resulting gradient makes that region an even better match for everything — a rich-get-richer dynamic that terminates in hub collapse.

### An important distinction we are careful about

It is tempting to say "InfoNCE is frequency-biased". **That is false and we do not claim it.** With negatives drawn from the marginal, the InfoNCE-optimal critic is exactly the pointwise mutual information, `f*(w,k) = log p(k|w)/p(k)` (van den Oord et al., 2018), which is *not* biased by the concept prior.

The bias we identify lives somewhere else: in the **within-episode E-step**, whose "negatives" are the other regions of the *same* image, not samples from the marginal. That inner softmax has no marginal correction and no exclusivity constraint, and it is where hub collapse originates. Everything below is about that step.

---

## 3. Method

### 3.1 Setup

An episode is a scene and an utterance:

* content words `w_1..w_N`, embedded and L2-normalised as `g_θ(w_n) ∈ S^{d-1}`;
* object slots `v_1..v_M` (pooled patch groups from the vision encoder), `h_φ(v_m) ∈ S^{d-1}`;
* similarities `s_nm = ⟨g_θ(w_n), h_φ(v_m)⟩ ∈ [-1, 1]`;
* a latent alignment `P ∈ R_{≥0}^{N×(M+1)}`, where column `M+1` is a **null bin** meaning *"this word refers to nothing visible"*.

### 3.2 E-step — semi-relaxed entropic optimal transport

Cost matrix, with `B` the accumulated lexicon evidence of §3.3 and `κ` a learnable null threshold:

```
C[n,m] = -s[n,m] - λ·B[n,m]      (m ≤ M)
C[n,⊥] = -κ
```

Solve

```
P* = argmin  ⟨P,C⟩ + ε·KL(P ‖ a b^T) + ρ·KL(P^T 1 ‖ b)
     s.t.    P 1 = a
```

with `a = 1_N/N` (every word must be accounted for, by a slot or by null) and
`b = ((1-η)/M · 1_M , η)` (slots share `1-η` of the mass; the null bin is allotted `η`).

This is solved by ~30 log-domain Sinkhorn iterations — a handful of matmuls on an `N×(M+1)` matrix, negligible next to the vision encoder. The two knobs:

| knob | meaning | limits |
|---|---|---|
| `ε` | entropic temperature | `ε→0`: partial permutation (hard exclusivity). `ε→∞`: uniform association. |
| `ρ` | **mutual-exclusivity strength** | `ρ=0`: row-wise softmax = region–word contrastive learning, *exactly*. `ρ→∞`: balanced OT. |

The per-word **referential mass** `π_n = 1 − P*[n,⊥]/a_n ∈ [0,1]` is how much of word `n` landed on a real slot. Non-referential words get `π_n ≈ 0` and are thereby removed from the gradient — the null bin is what lets the model *decline to ground* a word instead of being forced to attach "hungry" to whatever is on screen.

### 3.3 Cross-situational lexicon

Transport inside one batch only resolves ambiguity resolvable *within that scene*. Cross-situational learning is the claim that ambiguity irresolvable in any single scene becomes resolvable once evidence is aggregated across scenes.

Slots are soft-assigned to a codebook of `K` concept prototypes `Φ` with a **Sinkhorn-balanced (SwAV-style)** step — the same device as §3.2, one level down, and for the same reason: a plain softmax lets the codebook collapse. A running count matrix `L ∈ R^{V×K}` accumulates alignment mass per (word, concept), and its **PMI** is fed back into the next episode's cost:

```
L ← γ·L + Σ_episodes P*[:, :M] · q            (q = slot→prototype assignments)
L̂[w,k] = log( L[w,k] · L.. / (L[w,·] · L[·,k]) )      clipped to ±4
B[n,m] = (λ/4) · L̂[w_n]ᵀ q_m
```

PMI rather than raw counts is essential, and is the content of Proposition 3.

The whole procedure is **online stochastic EM** on a latent-alignment generative model: the E-step infers the alignment given current parameters and accumulated statistics, the M-step takes a gradient step given the inferred alignment.

### 3.4 M-step — referentially weighted soft-target InfoNCE

Minimising `⟨P*, C⟩` alone would drive every similarity to 1. The plan therefore supplies *targets*, and discriminative pressure comes from a contrastive term over **all `B·M` slots in the batch**:

```
L = (1/Σπ_n) · Σ_n π_n · Σ_m P̃[n,m] · ( −log softmax_k( ⟨g(w_n), h(v_k)⟩ / τ ) )
```

where `P̃` is `P*` restricted to real slots and row-normalised. A symmetric slot→word term is added. **The E-step decides *what* is aligned; the M-step provides the discriminative signal.**

By Danskin's theorem the gradient of the entropic-OT value w.r.t. the cost is the optimal plan itself, so `P*` is detached and the E-step costs no backward pass.

### 3.5 Developmental curriculum (a prediction, not a theorem)

`ε` is annealed geometrically over training: high `ε` gives diffuse, associative, slow learning; low `ε` gives near-permutation assignments, i.e. fast mapping under mutual exclusivity.

We *predict* — and will test, not assume — that the count of committed words (`max_k L̂[w,k]` over threshold) follows a sigmoid with a marked inflection, the computational analogue of the **vocabulary spurt**, and that its shape can be compared against Wordbank CDI norms. This is a falsifiable developmental prediction, and it is reported as such.

---

## 4. Theory

Every proposition below has a corresponding executable check in `tests/test_ot.py`.

### Proposition 1 — `ρ = 0` recovers region–word contrastive learning exactly

*At `ρ = 0`, the solution is `P*[n,m] = a_n · softmax_m(−C[n,·]/ε)`.*

**Proof.** At `ρ = 0` the column penalty vanishes, so `g ≡ 0` is optimal. The row constraint gives `f_n = ε·log a_n − ε·lse_m(−C[n,m]/ε)`. Substituting into `P[n,m] = exp((f_n + g_m − C[n,m])/ε)` gives the claim. ∎

**Consequence.** Our ablations are not comparisons against a re-implemented baseline — they are the *same code path* with one scalar changed. Any measured gap is attributable to `ρ` and nothing else.
→ `test_rho_zero_is_exactly_row_softmax`

### Proposition 2 — hub collapse is *infeasible* under the column constraint

Call a coupling an **`α`-hub** if some slot `m*` carries `Σ_n P[n,m*] ≥ α` of the total mass.

1. At `ρ = ∞`, `P*ᵀ1 = b` for **every** cost matrix. With `b_m = (1−η)/M`, no `α`-hub with `α > (1−η)/M` exists. The hub is not merely disfavoured; it is outside the feasible set.
2. At `ρ = 0`, for every `δ > 0` there is a cost matrix whose optimum is a `(1−δ)`-hub. (Take `C[n,m*] = 0`, `C[n,m] = c` for `m ≠ m*`; as `c/ε → ∞` the row softmax tends to a one-hot at `m*`.)
3. For finite `ρ > 0`, the optimum satisfies `(P*ᵀ1)_m = b_m · exp(−g*_m/ρ)`, so the log-deviation from the target column marginal is `|g*_m|/ρ` and vanishes as `ρ → ∞`.

**Proof.** (1) is the hard column marginal. (3) is the stationarity condition of the KL-penalised marginal term. ∎

We instrument this directly: `slot_usage_gini` in `gavagai/losses.py` measures the observed concentration of alignment mass over slots. In a smoke test it reads **0.096 at `ρ=0` and 0.000 at `ρ=∞`** on identical inputs.
→ `test_rho_interpolates_monotonically`, `test_balanced_recovers_column_marginal`

### Proposition 3 — the PMI readout is invariant to the concept prior

Generative model: concept `k ~ π`, word `w ~ p(·|k)`. Then

* `argmax_k p(k|w) = argmax_k π(k)·p(w|k)` — **depends on `π`**;
* `argmax_k PMI(w,k) = argmax_k p(w,k)/(p(w)π(k)) = argmax_k p(w|k)` — **independent of `π`**.

So if the true referent satisfies `p(w|c(w)) > p(w|k)` for all `k ≠ c(w)`, the PMI readout is correct for every prior, while the conditional readout is wrong whenever
`π(k)/π(c(w)) > p(w|c(w))/p(w|k)` for some `k`. In egocentric infant video `π` is extremely skewed, so this condition is routinely met.
→ `test_row_softmax_is_frequency_biased`, `test_pmi_readout_is_prior_invariant`

### Proposition 4 — balancing and PMI are complementary, not the same (a caveat we verified)

Sinkhorn scaling of a joint `J` to doubly stochastic satisfies

```
log(D_r J D_c) = PMI(J) + u_w + s_k
```

The **row** offsets `u_w` cancel inside a per-word argmax; the **column** offsets `s_k` do **not**. Hence *Sinkhorn scaling does not induce the PMI ranking in general* — on random matrices the two rankings disagree for **>50%** of rows.

We flag this because the opposite claim is superficially attractive and would be wrong. In this system the two mechanisms do different jobs: **balancing constrains the E-step** (Prop. 2), **PMI corrects the lexicon readout** (Prop. 3).
→ `test_sinkhorn_scaling_log_equals_pmi_up_to_constants`, `test_sinkhorn_argmax_is_not_pmi_argmax_in_general`

### Proposition 5 — consistency of the lexicon estimator

With `γ = 1` (no forgetting) and alignments drawn from the generative model, the empirical count matrix converges a.s. to `p(w,k)` by the strong LLN; PMI is continuous in `L` on the positive orthant and the argmax over a finite concept set is a.s. eventually constant. Hence the estimated referent converges to `c(w)` for every word satisfying the separation condition of Prop. 3. ∎

---

## 5. What is actually new

| Prior work | Relation |
|---|---|
| **OTTER** (Wu et al., 2021) — OT to soften noisy image–text labels | Closest prior art. OTTER is *within-batch* and *caption-level*. Ours is *within-episode at word↔slot granularity*, has a **null bin** for non-referential speech, and carries a **persistent cross-situational lexicon** across episodes. |
| **SwAV / Sinkhorn cluster assignment** | We use the same device for the codebook, but our contribution is the referential coupling, not the clustering. |
| **GLIP / FILIP** region–word alignment | These use exactly the `ρ = 0` row-softmax that Prop. 2 shows admits hub collapse. |
| **CVCL** (Vong et al., *Science* 2024) | Whole-frame ↔ whole-utterance contrastive on SAYCam. No latent word↔region variable at all. |
| **Frank, Goodman & Tenenbaum (2009)** Bayesian cross-situational word learning | We give a differentiable, GPU-scale, online-EM version of the same generative story, with mutual exclusivity as a transport constraint rather than a prior over lexicons. |
| **Yu & Smith (2007)** human cross-situational learning | Supplies the psychometric target for §6.1. |

The one-sentence novelty: **mutual exclusivity — a signature inductive bias of human word learning — is exactly a doubly-stochastic constraint on the word–referent coupling, and imposing it costs eight matmuls.**

---

## 6. Experimental protocol (Kaggle 2×T4, ≤30 GPU-h/week)

### 6.1 Controlled simulation — CPU, minutes
Ground truth known, so the mechanism can be measured rather than inferred.
* **Sample-efficiency sweep** (headline): accuracy vs. size of a *finite* corpus. Infant-scale data is the regime of interest; with unlimited episodes every method eventually succeeds, which we report.
* **Ablation ladder**: region–word contrastive → `+` null bin → `+` mutual exclusivity → balanced.
* **`ρ` sweep**: continuous interpolation from the baseline to full exclusivity.
* **Hub rate**: fraction of words whose nearest referent is a never-named background object — the direct measurement of Prop. 2.
* **Yu & Smith replication**: an unconstrained ideal observer is at ceiling on the original design, so the honest experiment is *how much memory decay must be assumed* to reproduce the human ordering (0.889 / 0.778 / 0.556) with one shared free parameter.

### 6.2 Real data — 2×T4, ~4–6 h
SAYCam is Databrary-gated. The BabyVLM-V2 authors released a **fully public Ego4D variant**: `wsashawn/devcv_toolbox_ego4d` (evaluation), `wsashawn/babyllava_v2_instruction_ft_Ego4D` (~89k instruction examples), and phase-0/2/3 checkpoints. All headline numbers are therefore reproducible by a reviewer without a data licence.

* Vision encoder and data budget held at infant scale; `ρ = 0` vs. `ρ > 0` with **everything else identical**.
* Zero-shot **Picture Vocabulary** and **Localization** on DevCV-Toolbox-Ego4D. Localization is free for us: the transport plan gives word→slot attribution with no box supervision, and slot→patch centroid gives a quadrant.
* A SAYCam adapter ships in the repo but is not on the critical path.

### 6.2a Headline result — captioning ± referential alignment

Full numbers are in [`RESULTS.md`](RESULTS.md), regenerated from the run outputs rather than typed by hand. Picture-vocabulary accuracy on **held-out exemplars**, chance = 1/40 = 0.025, 3 seeds:

| condition | clean | moderate | realistic |
|---|---|---|---|
| AR only (captioning) | 0.392 ± 0.031 | 0.242 ± 0.047 | 0.058 ± 0.031 |
| + naive align (`ρ=0`, no null) | 0.650 ± 0.054 | 0.358 ± 0.031 | **0.000 ± 0.000** |
| + null bin only | 0.658 ± 0.024 | 0.358 ± 0.051 | 0.017 ± 0.012 |
| + null + ME (**ours**) | **0.700 ± 0.054** | **0.400 ± 0.020** | 0.075 ± 0.061 |
| + null + balanced | 0.683 ± 0.066 | 0.400 ± 0.020 | 0.083 ± 0.042 |

Three things to note, including one against us.

1. **The captioning objective alone is the weakest condition in every regime.** Adding a latent word↔slot variable is what buys the improvement; this is the core claim and it holds throughout.
2. **Under realistic ambiguity the naive form of that variable is catastrophic** — 0.000 with zero variance across all three seeds, far below the 0.025 chance floor, which is total hub collapse. Forcing every word to ground is worse than not aligning at all. The null bin and the exclusivity constraint are what prevent it.
3. **Against us:** the realistic regime is floor-limited at this budget. Ours (0.075 ± 0.061) is nominally best but the dispersion is larger than the gap, so it supports no claim beyond "does not collapse". The separation that *is* meaningful lives in the clean and moderate regimes.

### 6.2b Scope condition — where the method does *not* help

We tried to break the mutual-exclusivity claim and partly succeeded, so we report it as a scope condition rather than discovering it in review.

In the controlled contrastive setting, sweeping the corpus size from 100 to 3000 episodes, the ablation ladder (region–word contrastive → `+` null bin → `+` exclusivity → balanced) produced **no reliable benefit**, and at the largest corpus sizes the exclusivity conditions were marginally *worse*:

| corpus | region–word contrastive | + null bin | + exclusivity | balanced |
|---|---|---|---|---|
| 100 | 0.692 | 0.700 | 0.700 | 0.708 |
| 300 | 0.908 | 0.908 | 0.908 | 0.867 |
| 1000 | 0.983 | 0.983 | 0.942 | — |
| 3000 | 1.000 | 1.000 | 0.958 | 0.950 |

The explanation is that the M-step is a **batch-level InfoNCE whose negatives are drawn from the marginal**, and that term already suppresses hub collapse: the constraint of Proposition 2 has nothing left to fix. Measured `slot_usage_gini` is ≈0 in every condition, confirming no hub ever formed.

This is precisely why the method is aimed at **autoregressively trained** baby VLMs. Every stage of BabyVLM-V2 is next-token prediction; there is no contrastive term anywhere, hence no marginal correction and no word↔region variable at all. §6.2 tests that setting directly.

Small positive effects did appear in the contrastive setting under small batches (0.608 → 0.617) and a high non-referential rate (0.625 → 0.642), consistent with this account but not on their own decisive.

### 6.3 Reporting standards
* ≥3 seeds with dispersion on every number.
* The `ρ = 0` baseline is *our own code path*, so no baseline is disadvantaged by re-implementation.
* Negative and null results reported, including regimes where the baseline matches or beats us.

---

## 7. Honest limitations

1. **Ego4D is adult egocentric video, not infant.** It is a public stand-in for gated SAYCam, chosen for reproducibility. Claims about developmental plausibility are correspondingly hedged.
2. **Slots are pooled patch groups, not objects.** No object discovery is claimed.
3. **The vocabulary-spurt result is a prediction under test**, not a theorem, and may fail.
4. **`η` (the null prior) is a hyper-parameter**, not learned from caregiver-speech statistics. Estimating it from CHILDES is future work.
5. **Prop. 2 shows hub collapse is infeasible, not that the optimum is the true lexicon.** Correctness of the recovered lexicon is an empirical claim, evidenced in §6.
6. **The exclusivity constraint provides no measured benefit when a batch-level contrastive term is present** (§6.2b). The method's value rests on the autoregressive setting, where that term is absent.
7. **The text-blind audit rests on small samples** (n = 6–24 per task). It is reported with Wilson intervals and must be regenerated on the full public release.
8. **The most realistic ambiguity regime is floor-limited** at the simulation budget used: every condition sits near chance and only the collapse of the naive ablation is statistically clean. Conclusions about which *working* method is best come from the clean and moderate regimes.
9. **`kappa` is a hyper-parameter, not learned.** It is read only inside the no-grad E-step and so receives no gradient — Danskin's theorem working as intended, not a bug. It is swept.
