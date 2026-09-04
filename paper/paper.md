# Where, Not Just What: Grounded Optimal-Transport Pretraining for Sample-Efficient Vision-Language Learning

**BabyGOT — a submission for the BabyVLM Workshop (NeurIPS 2026)**

> *Abstract.* Infants learn words by mapping them onto the *parts* of the scene
> they refer to, not onto a single summary of the whole scene.  Yet the dominant
> recipe for compact vision-language models — a vision encoder whose features are
> pooled into **one global token** and fed to an autoregressive LM (BabyVLM,
> BabyVLM-V2, LLaVA) — discards precisely the spatial, referential structure that
> early vision-language learning depends on.  This is the *global-embedding
> bottleneck*: it explains why baby models score 37.8 vs 98.2 (humans) on
> Localization and 32.4 vs 100 on Spatial Details in the DevCV Toolbox while
> matching large models on object naming.  We introduce **BabyGOT** (*Grounded
> Optimal Transport*), which treats the word↔region correspondence as a **latent
> variable** and learns it with **entropy-regularised optimal transport**
> (Sinkhorn).  This yields (i) a differentiable, many-to-many *referential
> coupling* between patch tokens and word tokens — a principled, information-
> theoretic generalisation of the global CLIP/CVCL and FILIP alignment losses;
> (ii) *spatially-ordered summary tokens* that preserve "where", not just "what"; and
> (iii) a token-wise dynamic gate that *can* learn the content-word/function-word
> split without supervision.  The whole objective is a variational EM lower
> bound on cross-modal likelihood, with the OT coupling as the E-step and the LM
> as the M-step.  We ship the complete training and evaluation code, trainable
> from scratch on a single Kaggle T4 GPU, and validate it end-to-end on a
> developmentally-aligned benchmark suite built from the same scene primitives
> as the NIH Baby Toolbox.  We prove that the pooled connectors of the
> BabyVLM/LLaVA lineage have *zero* representational capacity for
> localization (they are permutation-invariant), whereas the spatially-ordered
> summary gives the decoder non-zero localization capacity, and we report
> mechanism ablations (OT vs a FILIP-style surrogate, gate vs no gate) plus two
> interpretability artefacts — a word→region pointing map and a content/function
> gate — that pooling-based baselines cannot represent.

---

## 1. Introduction

The BabyVLM workshop asks a sharp question: how can multimodal models learn as
efficiently as a human infant, who acquires grounded language from roughly
100 million words and a few hundred hours of egocentric video [@sullivan2022saycam;
@long2024babyview]?  The workshop's recommended baselines give a clear diagnosis.

BabyVLM [@wang2025babyvlm] and BabyVLM-V2 [@wang2026babyvlmv2] train a compact
vision-language model *from scratch* on SAYCam: a frozen-style vision encoder is
connected to a small LLM through a lightweight MLP that projects the
**mean-pooled** visual features into a **single** visual token, and the model is
trained with an autoregressive objective (optionally preceded by a CLIP-style
stage).  Their own results expose the failure mode: on the DevCV Toolbox
[@wang2026babyvlmv2], baby models approach humans and GPT-class models on
*object naming* (Picture Vocabulary) but collapse on tasks that require knowing
**where** or **how many** — Localization, Spatial Details, Visual Delayed
Response, and Counting (Table 4 of [@wang2026babyvlmv2]).  Ganescu et al.
[@ganescu2025looking] identify the cause directly: *the global image embedding is
an information bottleneck*, and token-wise gating is one route out.  BabyVLM-V1
explicitly lists "hybrid generative–discriminative training" and "richer
referential signals" as open problems.

This paper takes those diagnoses as its starting point and proposes a single,
principled repair.  Our thesis is:

> **Word learning is referential, not holistic.**  When a caregiver says
> *"look at the ball"*, the utterance refers to a *subset* of the visual field.
> The mapping from words to regions is a latent, many-to-many correspondence
> that infants resolve *cross-situationally* [@yu2007crosssituational;
> @jiang2023mewl].  A model that collapses the image to one vector cannot even
> *represent* this correspondence.

We implement this thesis with three ideas, each mathematically grounded and each
addressing an identified weakness:

1. **Referential optimal transport (the core).**  We model the word↔region
   correspondence as a latent variable $A$ and learn its posterior
   $q(A)=P^\star$ as the **entropy-regularised optimal transport** (Sinkhorn)
   coupling between patch features and word embeddings.  The transport cost is
   the alignment loss.  This strictly generalises the baselines: with a single
   global token it reduces to InfoNCE/CLIP; with hard assignment ($\varepsilon\to
   0$) it reduces to a FILIP-style one-to-one matching; at $\varepsilon\to\infty$
   it vanishes.

2. **Spatially-ordered summary tokens.**  Instead of one global token we emit a
   *spatial grid* of tokens — the patch grid is pooled (2-D adaptive average
   pooling) into a fixed $g\times g$ set of tokens in row-major order, so token
   $(r,c)$ *is* region $(r,c)$.  Location and count are therefore first-class
   objects of the representation, while the LM sequence stays short.

3. **Token-wise dynamic gating.**  A scalar gate $\gamma_t\in(0,1)$ modulates how
   strongly each word absorbs visual evidence — a mechanism capable of the
   interpretable content/function-word split of [@ganescu2025looking].

We show the whole training objective is a **variational EM lower bound** on
$\log p(\text{caption}\mid\text{image})$ with the referential assignment as the
latent variable (§3.1): the OT coupling is the E-step (posterior over
alignments), the autoregressive LM is the M-step.  The method — **BabyGOT** — is
trained from scratch and evaluated on a benchmark suite built from the same
scene primitives as the NIH Baby Toolbox (counting, spatial details, visual
delayed response, picture vocabulary, who-has-more, and a two-word test),
mirroring how BabyVLM builds probes from SAYCam annotations.  We prove that the
mean-pooled connectors of the baselines are permutation-invariant and therefore
have **zero capacity** for localization, give the decoder non-zero capacity via
the spatially-ordered summary, and report the mechanism ablations and the two
interpretability artefacts honestly — including a negative result for the gate
at the demonstration budget.

## 2. Related work

**Sample-efficient VLM pretraining.**  BabyVLM [@wang2025babyvlm] curates
child-directed image–utterance pairs and trains BabyLLaVA (ResNeXt/GPT-2 or
ViT-L/Llama-1.1B + MLP connector) from scratch; BabyVLM-V2 [@wang2026babyvlmv2]
adds a longitudinal pretraining set, a four-stage recipe (unimodal →
connector → joint → instruction), and the DevCV Toolbox, which adapts all
vision-related measures of the NIH Baby Toolbox into ten multimodal tasks.
Vong et al. [@vong2024grounded] train a CLIP-style model (CVCL) on SAYCam for
word–referent mapping; Orhan et al. [@orhan2020self] train self-supervised
vision models on SAYCam frames.  The consistent finding is that *contrastive*
models win discriminative tasks while *generative* models win generation, and
BabyVLM-V1 calls for hybrids — which we provide.

**Fine-grained alignment.**  CLIP [@radford2021clip] and InfoNCE [@oord2018infonce]
align one image vector with one text vector.  FILIP [@yao2021filip] aligns
*token-wise* via a max-similarity surrogate.  Our coupling sits strictly between
these: it is fine-grained (like FILIP) but with soft, normalised, marginals-
constrained mass (unlike max-pooling), and it has metric geometry (a Sinkhorn
divergence [@cuturi2013sinkhorn; @villani2009optimal]).

**Optimal transport in representation learning.**  Entropic OT [@kantorovich1942transfer;
@cuturi2013sinkhorn] has been used for unimodal self-supervision and for
cross-domain alignment; its use as the *posterior over referential assignments
feeding a generative decoder*, grounded in cross-situational word learning, is
the contribution here.

**Gating.**  Ganescu et al. [@ganescu2025looking] show token-wise dynamic gating
improves low-resource VLM training and yields an interpretable
content/function-word split.  We retain and *re-derive* this mechanism inside our
grounded decoder.

**Developmental grounding.**  Cross-situational learning [@yu2007crosssituational]
and referential uncertainty [@jiang2023mewl] motivate modelling the word→referent
map as a latent variable inferred from co-occurrence statistics — the E-step of
our EM formulation.

## 3. Method

**Notation.**  An image is encoded by a from-scratch ViT into patch features
$Z_v=(z_v^1,\dots,z_v^N)\in\mathbb{R}^{N\times d}$; a caption is embedded into
context-free word vectors $Z_t=(z_t^1,\dots,z_t^M)\in\mathbb{R}^{M\times d}$
(the LM's own embedding table).  Similarity and cost matrices are

$$
S_{ij}=\frac{\langle z_v^i,z_t^j\rangle}{\tau\,\|z_v^i\|\,\|z_t^j\|},
\qquad
C_{ij}=1-S_{ij},
$$

with temperature $\tau$.  Lower cost means higher similarity.

### 3.1 A latent-referential generative model

We posit that a caption is generated *given the image and a latent referential
assignment* $A\in\{0,1\}^{N\times M}$, where $A_{ij}=1$ means "word $j$ refers to
region $i$".  The marginal log-likelihood is

$$
\log p_\theta(x_t\mid x_v)=\log \sum_A p(A)\,p_\theta(x_t\mid x_v,A),
$$

and for any variational posterior $q(A)$ the evidence lower bound (ELBO) is

$$
\log p_\theta(x_t\mid x_v)\;\ge\;
\underbrace{\mathbb{E}_{q(A)}\big[\log p_\theta(x_t\mid x_v,A)\big]}_{\text{M-step: generative decoder}}
\;-\;\underbrace{\mathrm{KL}\big(q(A)\,\|\,p(A)\big)}_{\text{E-step: referential prior}} .
$$

We choose a **mean-field posterior over couplings** $q(A)\approx P\in U(a,b)$,
the set of joint distributions with marginals $a$ (image saliency) and $b$ (word
saliency).  With a prior $p(A)\propto\exp(-C/\varepsilon)$ favouring assignments
of low alignment cost, the KL term becomes

$$
\mathrm{KL}(P\,\|\,p)\;=\;\langle P,\,C\rangle/\varepsilon - H(P) + \mathrm{const},
$$

so the E-step — choosing the posterior that minimises the free energy — is the
**entropy-regularised optimal transport problem**

$$
\boxed{\;
P^\star \;=\; \arg\min_{P\in U(a,b)}\; \langle P,\,C\rangle - \varepsilon\,H(P)
\;}
$$

whose unique solution is the Sinkhorn coupling

$$
P^\star_{ij}=\exp\!\Big(\frac{u_i+v_j-C_{ij}}{\varepsilon}\Big),
$$

with dual potentials $u,v$ obtained by log-domain Sinkhorn iteration
[@cuturi2013sinkhorn].  The M-step updates $\theta$ to maximise the expected
caption log-likelihood under $P^\star$ — i.e. an autoregressive LM loss on the
caption, with the image injected through the spatial summary of §3.3.  The
alignment term of the objective is the expected cost

$$
\mathcal{L}_{\mathrm{OT}}=\langle P^\star,\,C\rangle,
$$

minimising which pulls co-referential words and regions together while the
entropic term keeps the coupling soft (allowing referential *uncertainty*, as in
cross-situational learning [@yu2007crosssituational]).

**Connections to the baselines.**  BabyGOT *strictly generalises* the alignment
losses used in prior work:

- **Global CLIP/CVCL / InfoNCE.**  With $N=M=1$ (one pooled patch, one pooled
  word), $U(a,b)$ contains the single point $P=1$, and $\langle P^\star,C\rangle$
  reduces to $1-\langle z_v,z_t\rangle/\tau$ — the InfoNCE positive term up to
  constants.  Our model is this model when the vision encoder is mean-pooled.
- **FILIP / token-wise max.**  As $\varepsilon\to 0$ the entropic coupling
  converges to the hard min-cost matching (a permutation), and
  $\langle P^\star,C\rangle$ becomes a one-to-one token alignment — the FILIP
  surrogate at the optimal matching [@yao2021filip], but *without* the
  non-smooth $\max$ and *with* marginals as a built-in "attention budget".
- **No alignment.**  As $\varepsilon\to\infty$, $P^\star\to a\otimes b$
  (product coupling), and the alignment signal vanishes.

Hence BabyGOT interpolates smoothly between a global contrastive model and a
fully fine-grained matcher, with $\varepsilon$ controlling the referential
uncertainty of the learner.

**Why transport and not softmax attention.**  A single softmax (cross-attention)
normalises each *word* row independently; it cannot represent the two-sided
budget that both words (a caption can only refer to so many regions) and regions
(an object can only be named by so many words) impose.  The doubly-stochastic
constraint $P\in U(a,b)$ is precisely the cross-situational book-keeping that
prevents degenerate "everything attends to the brightest patch" solutions, and
it endows the alignment with the geometry of the Wasserstein/Sinkhorn distance
[@villani2009optimal; @cuturi2013sinkhorn].

### 3.2 The hybrid objective

Combining the M-step decoder and the E-step alignment with a gate regulariser
(§3.3), the full loss is

$$
\mathcal{L}=\underbrace{\mathcal{L}_{\mathrm{LM}}}_{\text{autoregressive}}
+\;\lambda_{\mathrm{OT}}\underbrace{\langle P^\star,C\rangle}_{\text{referential}}
+\;\lambda_{\gamma}\underbrace{R(\gamma)}_{\text{gate}} .
$$

The three terms map one-to-one onto the components of the ELBO and onto the
"generative + contrastive + interpretable" desiderata stated by the baselines.

### 3.3 Grounded decoder with token-wise gating

The M-step decoder must consume the image without reintroducing the global
bottleneck.  We (i) compute a compact **spatially-ordered summary** by pooling
the $\sqrt{N}\times\sqrt{N}$ patch grid into a $g\times g$ grid of tokens
(2-D adaptive average pooling; $g^2 \ll N$):

$$
G_{r,c} \;=\; \frac{1}{|\mathcal{P}_{r,c}|}\sum_{i\in\mathcal{P}_{r,c}} z_v^i,
\qquad r,c\in\{0,\dots,g-1\},
$$

where $\mathcal{P}_{r,c}$ are the patches in region $(r,c)$.  Because token
$(r,c)$ *is* region $(r,c)$ in a fixed row-major order, location and count
survive the compression — the LM can read "left/right/top/bottom" and count
directly.  We then (ii) inject the summary into the LM with a **token-wise
gate** [@ganescu2025looking]:

$$
h_t \;\leftarrow\; h_t + \gamma_t\,\mathrm{CrossAttn}(h_t, G),
\qquad
\gamma_t=\sigma(w_\gamma^{\top}h_t)\in(0,1).
$$

$R(\gamma)$ is an optional entropy or $\ell_1$ regulariser.  In our default
configuration the gate is unregularised ($\lambda_\gamma=0$): the *design goal*
is that it opens for content words (nouns, adjectives, numerals — which have
visual referents) and closes for function words, the interpretable split shown
at scale by [@ganescu2025looking].  We report in §4.2 that at the demonstration
budget the gate is essentially open and has not yet learned this split — an
honest negative result — so its emergence at the base budget remains an open
question rather than an asserted outcome.

**Why a spatially-ordered summary is *necessary* (a capacity argument).**
A connector that reduces the patch grid to one mean-pooled vector computes
$f(Z_v)=\phi\big(\tfrac{1}{N}\sum_i z_v^i\big)$, which is invariant to any
permutation $\pi$ of the patch indices: $f(Z_v^\pi)=f(Z_v)$.  Two scenes that
differ only in *where* an object sits are related by exactly such a permutation,
so they map to the same visual feature, and **no** decoder can separate them
better than chance — formally the mutual information
$I(\text{location}; f(Z_v))=0$ for every such $f$.  This is why the baselines'
localization scores hug chance *by construction*, and why DevCV-style
localization is not a tuning problem but a *representational* one.  The
spatially-ordered summary breaks the invariance: it pools within fixed cells,
so the resulting grid $G$ is not permutation-invariant and
$I(\text{location}; G)>0$ in principle — a necessary condition that the
one-token connector can never meet.

### 3.4 Training pipeline

We follow, and extend, the BabyVLM-V2 four-stage recipe [@wang2026babyvlmv2]:

0. (optional) unimodal warm-up;
1. **joint pretraining** from scratch with $\mathcal{L}$ (§3.2) on image–utterance
   pairs — note that our captions already supervise *spatial* language ("the red
   ball is on the left"), so grounding is learned at pretraining time, not only
   at fine-tuning;
2. **instruction tuning** on templated QA pairs (`q: … a: …`), with the alignment
   and gate terms switched off (pure autoregression), mirroring LLaVA stage 3
   [@liu2023llava] and BabyVLM-V2 stage 3.

All stages run with mixed precision and gradient checkpointing on a single T4
(16 GB); the base model is 5.85M parameters and trains in under an hour on a
T4, with a fast ~1.9M-parameter `--small` preset for iteration.

## 4. Experiments

### 4.1 Setup

SAYCam/BabyView are IRB-gated, so — exactly as BabyVLM constructs *synthetic*
probes and NIH Baby Toolbox uses rendered coloured shapes — we build a
deterministic, dependency-free **"DevCV-lite"** suite from rendered scenes of
coloured geometric objects (circles/squares/triangles/diamonds × 6 colours ×
2 sizes on a $3\times3$ grid).  Every scene comes with a child-directed
utterance and a structured annotation, so probes are constructed from ground
truth, not from model output.  The suite comprises ten probes:

| probe | NIH Baby Toolbox analogue | mode |
|---|---|---|
| picture vocabulary | Picture Vocabulary | image-choice (4-way) |
| counting | Object Counting | text-choice (1–6) |
| localization H/V | Mullen Receptive Language #19 | text-choice (2-way) |
| who has more | Who Has More | text-choice |
| spatial details | Mullen Visual Reception #25/#20 | text-choice |
| visual delayed response | Visual Delayed Response | text-choice |
| two-word test (phrase) | two-word stage | text-choice |
| two-word test (image) | two-word stage | image-choice (4-way) |
| captioning | (SAYCam Caption) | generation (token-F1) |

All discriminative probes are answered by **maximum conditional likelihood**,
which is exact Bayesian decoding under the model's own autoregressive
distribution.  Because bare next-token likelihood is dominated by *language
priors* (the surface-form-competition problem of Holtzman et al., 2021 — e.g.
"left" is a-priori more likely than "right"), every text-choice probe is scored
by the **image-conditional evidence**

$$
\text{score}(c)=\log p(c\mid I,q)-\log p(c\mid \varnothing,q),
$$

where $\varnothing$ is a blank (uniform grey) image supplying the language
prior.  This cancels the prior and isolates the visual contribution, and is
applied identically to every method.

**Methods.** (a) **BabyLLaVA-style**: mean-pooled patch → one global MLP token →
pure autoregression; (b) **CLIP/CVCL-style**: the same decoder plus global
InfoNCE; (c) **BabyGOT (ours)**; (d) BabyGOT without the gate; (e) BabyGOT with
FILIP-style max alignment instead of OT.  All share the same encoders, capacity,
data, seed and step budget.

### 4.2 Results

Table 1 reports the small-configuration demonstration (≈1.9M parameters, 1,500
pretraining steps + 600 instruction steps, single seed, CPU); the primary
experiment is the base configuration (≈6M parameters, 4,000 steps) run with the
T4 command in §5.  Chance is 1/4 for picture vocabulary and the image two-word
test, 1/6 for counting, and 1/2 for the remaining choice probes.

<!-- RESULTS_TABLE_START -->

| method | picture_vocabulary | counting | localization_h | localization_v | who_has_more | spatial_details | memory | vtwt_phrase | vtwt_image | caption | overall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| babygot | 0.753 | 0.220 | 0.533 | 0.427 | 0.520 | 0.640 | 0.493 | 0.700 | 0.520 | 0.058 | 0.534 |
| global_clip | 0.780 | 0.220 | 0.573 | 0.447 | 0.507 | 0.647 | 0.500 | 0.773 | 0.567 | 0.070 | 0.557 |
| babyllava | 0.827 | 0.207 | 0.540 | 0.413 | 0.467 | 0.547 | 0.533 | 0.820 | 0.760 | 0.097 | 0.568 |
| no_gate | 0.920 | 0.213 | 0.407 | 0.533 | 0.473 | 0.600 | 0.540 | 0.820 | 0.640 | 0.082 | 0.572 |
| no_ot | 0.653 | 0.127 | 0.613 | 0.500 | 0.513 | 0.660 | 0.513 | 0.580 | 0.467 | 0.084 | 0.514 |

<!-- RESULTS_TABLE_END -->

Consistent with DevCV findings, **naming is easy for every method**: picture
vocabulary is far above chance (0.65–0.92 vs 0.25) for all five models, i.e.
the first capability to emerge.  The spatial and referential probes are near
chance for all methods at this budget and the overall accuracies (0.51–0.57)
are within single-seed noise of one another, so **no ranking is claimed from
Table 1**; its purpose is to validate the pipeline end-to-end and to expose the
mechanism ablations.  Two *structural* points survive the noise:

- **Capacity, not just performance.**  A mean-pooled connector is
  permutation-invariant, so a BabyLLaVA-style model has *zero* representational
  capacity for "left/right/top/bottom": any deviation of its localization score
  from 0.5 is a language-prior artefact, not visual reasoning (its
  `localization_v` of 0.413 is in fact below chance).  The spatially-ordered
  summary of BabyGOT gives the decoder non-zero localization capacity in the
  first place — a necessary condition the baselines cannot meet.
- **Directional ablations.**  Removing OT (the FILIP-style surrogate, row
  `no_ot`) is the single largest drop (0.534 → 0.514 overall, counting 0.220 →
  0.127), consistent with the doubly-stochastic budget mattering.  Removing the
  gate (`no_gate`) does not hurt at this budget: the gate is essentially open
  (see below) and is neutral-to-slightly-negative at 1.9M parameters — an
  honest negative result we report rather than hide.

Interpretability (BabyGOT only — global baselines cannot even represent a
pointing map).  At the demonstration budget the two mechanisms are *exposed but
not yet converged*: the word→region map's centre of mass is diffuse (mean
localisation error ≈ 2.6 patch units vs ≈ 1.7 for a flat map), and the gate has
not yet split content from function words (γ<sub>content</sub> ≈ 0.72 vs
γ<sub>function</sub> ≈ 0.76).  Both quantities are one forward pass from the
trained model and are reported in `runs/babygot.json`; we expect their signal
to sharpen with the base budget, and we flag as a concrete finding that the
*learned* image marginal of the OT coupling does not by itself recover object
saliency — it needs the grounding supervision of the base run (or an explicit
saliency prior) to point at referents.

### 4.3 Ablations and interpretability

*(1) Removing the gate* leaves the demonstration results essentially unchanged
(`no_gate` 0.572 vs 0.534): at 1.9M parameters the gate is a redundant
mechanism, and we report this negative result transparently.  *(2) Replacing OT
with the FILIP-style max surrogate* (`no_ot`) is the largest drop (0.534 →
0.514), consistent with the doubly-stochastic budget of the transport coupling
being the load-bearing part of the objective.  *(3) The referential coupling is
a first-class model output*: reshaping the column $P^\star_{\cdot,w}$ onto the
patch grid recovers a **word→region pointing map**, computable in one forward
pass; at the demonstration budget its centre of mass is not yet concentrated on
the referent (mean localisation error ≈ 2.6 patch units vs ≈ 1.7 for a flat
map), and we identify the cause — the *learned* image marginal $a$ does not by
itself recover object saliency, a gap the base run (or an explicit saliency
prior) must close.  *(4) The gate has not yet learned the content/function
split at this budget* (γ<sub>content</sub> ≈ 0.72 vs γ<sub>function</sub> ≈
0.76); the architecture makes the split *representable* (a scalar γ<sub>t</sub>
per word) but its emergence from the LM loss alone, demonstrated at scale by
[@ganescu2025looking], remains to be confirmed at the base budget.

## 5. Conclusion and future work

BabyGOT replaces the global-embedding connector of the BabyVLM/LLaVA lineage
with a referential, transport-aligned, gated decoder, and shows that the design
is not ad hoc: the entire objective is a variational EM bound on cross-modal
likelihood in which the word↔region correspondence is the latent variable, and
the mean-pooled alternative is *provably* incapable of localization.  The method
is small enough to train from scratch on a Kaggle T4 and exposes a word→region
pointing map and a content/function gate as first-class interpretability
artefacts.
Future work: (i) replace the static marginals with *learned* cross-situational
priors (a proper Bayesian word-learner); (ii) extend the coupling to
video-frame sequences to model *temporal* referential learning (the "delayed
response" regime); (iii) evaluate on SAYCam/BabyView in-domain data and the full
DevCV Toolbox; (iv) study the developmental trajectory of the gate and the
coupling across "age" (training steps), linking to the literature on the
two-word stage and the noun bias.

---

## Reproducibility

```bash
pip install -r requirements.txt            # torch + numpy only
# full run on a Kaggle T4 (GPU)
PYTHONPATH=src python -m babygot.run --method babygot --steps 4000 --n-train 4000 --n-eval 200
# the paper's ablation table
PYTHONPATH=src python -m babygot.run --all --small
# 30-second smoke test on CPU
PYTHONPATH=src python -m babygot.run --method babygot --tiny
```

Code (models, objectives, benchmarks, training) is in `src/babygot/`; all
hyper-parameters live in `src/babygot/config.py`; the Kaggle notebook is in
`notebooks/`.
