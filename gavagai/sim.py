"""Controlled cross-situational word-learning environment.

This module isolates the mechanism claimed in ``METHOD.md`` from every confound
of real vision.  A learner sees a stream of *ambiguous* trials: several objects
are present and several words are heard, with no indication of which word goes
with which object (Quine's ``gavagai`` problem).

Two properties of real egocentric infant video are modelled explicitly because
they are what breaks standard region-word contrastive learning:

**Background objects.**  Hands, floor and table are present in almost every
frame but are almost never named.  They are the raw material of a *hub*: a slot
that is available to absorb every word's alignment mass on every trial.

**Non-referential speech.**  Most word tokens a caregiver produces ("look at
that", "are you hungry") refer to nothing currently visible.

Learners come in two flavours.  ``CrossSituationalLearner`` is a count-based
ideal observer, useful for psychometric comparison against human data.
``run_embedding_learner`` trains real embeddings with the real training
objective from ``gavagai.losses``, so an ablation over ``rho`` changes exactly
one thing and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import gavagai_loss
from .ot import referential_plan


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


@dataclass
class WorldConfig:
    """Generative parameters of the referential world."""

    n_words: int = 18
    """Number of nameable (foreground) objects.  Ground truth is word i <-> object i."""

    objects_per_trial: int = 4
    """Foreground objects visible per trial (Yu & Smith's 2x2 / 3x3 / 4x4)."""

    freq_skew: float = 0.0
    """Zipf exponent on foreground object sampling.  0.0 = uniform."""

    null_rate: float = 0.0
    """Probability that a heard word is filler rather than the object's name."""

    n_filler_words: int = 12
    """Size of the non-referential filler vocabulary."""

    noise_rate: float = 0.0
    """Probability a referential word is replaced by the wrong object's name."""

    n_background: int = 0
    """Objects present in nearly every scene but almost never named."""

    background_prob: float = 0.9
    """Probability a given background object appears in a trial."""

    background_name_prob: float = 0.02
    """Probability a present background object actually gets named."""

    n_exemplars: int = 1
    """Distinct appearances per object category seen during training.

    With one exemplar per object the task degenerates: a word only has to match
    a single fixed vector, and every method solves it.  Real word learning is
    *category* learning -- a word must cover many varied instances and
    generalise to unseen ones -- which is what makes referential noise costly."""

    n_eval_exemplars: int = 4
    """Held-out appearances per category, used only at readout."""

    within_spread: float = 0.6
    """Appearance variation within a category, relative to the category centroid."""

    absent_ref_prob: float = 0.0
    """Probability of uttering a real object's name while that object is NOT in
    the scene ("ball!" as the ball rolls out of view).  Vong et al. report the
    target referent is visible for only a minority of caregiver utterances, so
    this is the dominant realistic source of *false* word-object co-occurrence:
    the word is forced onto whatever happens to be visible, which is usually a
    never-named background object."""

    n_feature_families: int = 0
    """If > 0, object appearance vectors are drawn around this many shared family
    centroids, making objects visually confusable.  Real object categories are
    not orthogonal, and orthogonal prototypes make the task trivial."""

    family_spread: float = 0.7
    """How far an object's appearance sits from its family centroid."""

    seed: int = 0

    @property
    def n_objects(self) -> int:
        return self.n_words + self.n_background

    @property
    def vocab_size(self) -> int:
        return self.n_words + self.n_filler_words


class ReferentialWorld:
    """Samples ambiguous ``(words, objects)`` trials from a known ground-truth lexicon.

    Object indices ``[0, n_words)`` are nameable foreground objects; indices
    ``[n_words, n_objects)`` are background.  ``truth[w] = w``.
    """

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n_objects = cfg.n_objects
        self.vocab_size = cfg.vocab_size
        self.truth = np.arange(cfg.n_words)
        ranks = np.arange(1, cfg.n_words + 1, dtype=np.float64)
        p = ranks ** (-cfg.freq_skew)
        self.object_prior = p / p.sum()

    def sample_corpus(self, n_episodes: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Draw a *finite* corpus once, the way a child gets a finite amount of input.

        Sample efficiency is the quantity of interest, so the headline sweep
        varies ``n_episodes`` and trains to convergence on each fixed corpus.
        """
        return [self.sample_trial() for _ in range(n_episodes)]

    def appearance_prototypes(self, feat_dim: int, generator=None) -> "torch.Tensor":
        """Category centroid for every object, optionally clustered into families."""
        cfg = self.cfg
        if cfg.n_feature_families <= 0:
            return F.normalize(torch.randn(self.n_objects, feat_dim, generator=generator), dim=-1)
        fams = F.normalize(
            torch.randn(cfg.n_feature_families, feat_dim, generator=generator), dim=-1
        )
        assign = torch.arange(self.n_objects) % cfg.n_feature_families
        jitter = torch.randn(self.n_objects, feat_dim, generator=generator)
        return F.normalize(fams[assign] + cfg.family_spread * jitter, dim=-1)

    def appearance_bank(self, feat_dim: int, generator=None) -> "torch.Tensor":
        """``(n_objects, n_exemplars + n_eval_exemplars, feat_dim)`` appearance bank.

        Training may only use the first ``n_exemplars`` slices; the remainder are
        held out, so the readout measures generalisation to unseen instances of
        a category rather than recall of a single stored vector.
        """
        cfg = self.cfg
        centroids = self.appearance_prototypes(feat_dim, generator=generator)
        total = cfg.n_exemplars + cfg.n_eval_exemplars
        jitter = torch.randn(self.n_objects, total, feat_dim, generator=generator)
        return F.normalize(centroids.unsqueeze(1) + cfg.within_spread * jitter, dim=-1)

    def sample_trial(self) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        k = min(cfg.objects_per_trial, cfg.n_words)
        fg = self.rng.choice(cfg.n_words, size=k, replace=False, p=self.object_prior)

        bg = [
            cfg.n_words + j
            for j in range(cfg.n_background)
            if self.rng.random() < cfg.background_prob
        ]

        words = []
        for obj in fg:
            if self.rng.random() < cfg.null_rate:
                words.append(cfg.n_words + int(self.rng.integers(cfg.n_filler_words)))
            elif self.rng.random() < cfg.noise_rate:
                words.append(int(self.rng.integers(cfg.n_words)))
            else:
                words.append(int(self.truth[obj]))
        for _ in bg:
            if self.rng.random() < cfg.background_name_prob:
                words.append(cfg.n_words + int(self.rng.integers(cfg.n_filler_words)))

        if cfg.absent_ref_prob > 0:
            present = set(int(o) for o in fg)
            for _ in range(len(fg)):
                if self.rng.random() < cfg.absent_ref_prob:
                    absent = [i for i in range(cfg.n_words) if i not in present]
                    if absent:
                        words.append(int(self.rng.choice(absent)))

        words = np.array(words, dtype=np.int64)
        objects = np.array(list(fg) + bg, dtype=np.int64)
        self.rng.shuffle(words)
        return words, objects


# ---------------------------------------------------------------------------
# Count-based ideal observer (psychometrics)
# ---------------------------------------------------------------------------


@dataclass
class LearnerConfig:
    eps: float = 0.05
    rho: float | None = None
    null_prior: float = 0.3
    kappa: float = 0.0
    gamma: float = 1.0
    """Fraction of accumulated lexicon evidence *retained* per trial.

    1.0 = perfect memory.  Lower values model forgetting, and this is the single
    free parameter fitted to human Yu & Smith accuracy.  Matches the retention
    convention in ``gavagai.lexicon`` (multiply by gamma), which an earlier
    version of this file inverted."""
    feedback: float = 1.0
    smoothing: float = 0.1
    use_null: bool = True
    readout: str = "pmi"

    attend_k: int | None = None
    """Maximum objects (and words) the learner actually encodes on a trial.

    An unconstrained ideal observer solves Yu & Smith's design at ceiling for
    every memory-decay rate, so forgetting is the wrong capacity limit.  The
    account that does predict the human pattern is limited *encoding*: a learner
    registers only a couple of the word-object pairs available on each trial
    (Trueswell et al.'s propose-but-verify; Yurovsky & Frank).  This is the one
    free parameter fitted to human accuracy."""

    ot_iters: int = 60


class CrossSituationalLearner:
    """Online stochastic EM over the latent word-referent alignment.

    E-step: solve semi-relaxed entropic OT for the current trial using the
    accumulated PMI lexicon as the cost.  M-step: accumulate the soft alignment.
    """

    def __init__(self, vocab_size: int, n_objects: int, cfg: LearnerConfig):
        self.cfg = cfg
        self.counts = torch.zeros(vocab_size, n_objects, dtype=torch.float64)
        self.n_trials = 0
        self._rng = np.random.default_rng(0)

    def pmi(self) -> torch.Tensor:
        c = self.counts + self.cfg.smoothing
        return torch.log(c * c.sum() / (c.sum(1, keepdim=True) * c.sum(0, keepdim=True)))

    def scores(self) -> torch.Tensor:
        if self.cfg.readout == "conditional":
            c = self.counts + self.cfg.smoothing
            return torch.log(c / c.sum(0, keepdim=True))
        return self.pmi()

    def observe(self, words: np.ndarray, objects: np.ndarray) -> None:
        if self.cfg.attend_k is not None:
            k = self.cfg.attend_k
            if len(words) > k:
                words = words[self._rng.choice(len(words), k, replace=False)]
            if len(objects) > k:
                objects = objects[self._rng.choice(len(objects), k, replace=False)]
        w = torch.as_tensor(words, dtype=torch.long)
        o = torch.as_tensor(objects, dtype=torch.long)
        sim = (self.cfg.feedback * self.pmi()[w][:, o]).to(torch.float32).unsqueeze(0)
        plan, _ = referential_plan(
            sim,
            kappa=self.cfg.kappa if self.cfg.use_null else -1e4,
            eps=self.cfg.eps,
            rho=self.cfg.rho,
            null_prior=self.cfg.null_prior if self.cfg.use_null else 0.0,
            n_iter=self.cfg.ot_iters,
        )
        plan = plan[0].to(torch.float64) * len(w)
        if self.cfg.gamma < 1.0:
            self.counts.mul_(self.cfg.gamma)
        self.counts[w.unsqueeze(1), o.unsqueeze(0)] += plan
        self.n_trials += 1

    def predict(self, n_real_words: int) -> torch.Tensor:
        return self.scores()[:n_real_words, : n_real_words].argmax(1)


def run_learner(world: ReferentialWorld, cfg: LearnerConfig, n_trials: int) -> dict:
    learner = CrossSituationalLearner(world.vocab_size, world.n_objects, cfg)
    curve = []
    every = max(1, n_trials // 40)
    for t in range(n_trials):
        learner.observe(*world.sample_trial())
        if (t + 1) % every == 0:
            pred = learner.predict(world.cfg.n_words).numpy()
            curve.append((t + 1, float((pred == world.truth).mean())))
    pred = learner.predict(world.cfg.n_words).numpy()
    return {"accuracy": float((pred == world.truth).mean()), "curve": curve, "learner": learner}


# ---------------------------------------------------------------------------
# Embedding learner: the faithful ablation
# ---------------------------------------------------------------------------


@dataclass
class EmbedConfig:
    dim: int = 64
    feat_dim: int = 128
    feat_noise: float = 0.25
    steps: int = 800
    batch: int = 64
    lr: float = 3e-3
    tau: float = 0.07
    eps: float = 0.05
    rho: float | None = 1.0
    kappa: float = 0.0
    null_prior: float = 0.5
    use_null: bool = True
    eval_every: int = 100
    shared_word_encoder: bool = False
    """If True, words are random sparse feature bags passed through a shared MLP
    instead of getting a private embedding row.  A private table lets filler
    words be learned in isolation, so noise cannot propagate between words; real
    text encoders share parameters and do propagate it."""

    word_feat_dim: int = 96
    word_feat_active: int = 6
    seed: int = 0


class _ToyEncoders(nn.Module):
    def __init__(self, vocab: int, feat_dim: int, dim: int, word_feats: torch.Tensor | None = None):
        super().__init__()
        self.vis = nn.Sequential(nn.Linear(feat_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        if word_feats is None:
            self.word = nn.Embedding(vocab, dim)
            nn.init.normal_(self.word.weight, std=0.02)
            self.word_mlp = None
        else:
            self.register_buffer("word_feats", word_feats)
            self.word = None
            self.word_mlp = nn.Sequential(
                nn.Linear(word_feats.shape[1], dim), nn.GELU(), nn.Linear(dim, dim)
            )

    def words(self, idx):
        if self.word is not None:
            return F.normalize(self.word(idx), dim=-1)
        return F.normalize(self.word_mlp(self.word_feats[idx]), dim=-1)

    def slots(self, feats):
        return F.normalize(self.vis(feats), dim=-1)


def _pad(seqs: list[np.ndarray], device) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(s) for s in seqs)
    out = np.zeros((len(seqs), n), dtype=np.int64)
    mask = np.zeros((len(seqs), n), dtype=bool)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
        mask[i, : len(s)] = True
    return (
        torch.as_tensor(out, device=device),
        torch.as_tensor(mask, device=device),
    )


def run_embedding_learner(
    world: ReferentialWorld,
    cfg: EmbedConfig,
    corpus: list | None = None,
) -> dict:
    """Train word/object embeddings on ambiguous trials with the real objective.

    If ``corpus`` is given, training samples with replacement from that fixed
    finite set of episodes -- the infant-scale regime.  Otherwise fresh episodes
    are drawn on every step (the infinite-data regime, where the problem is easy
    and every method eventually succeeds).
    """
    torch.manual_seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(cfg.seed + 1)
    rng = np.random.default_rng(cfg.seed + 2)

    bank = world.appearance_bank(cfg.feat_dim, generator=g).to(dev)
    n_tr = world.cfg.n_exemplars
    word_feats = None
    if cfg.shared_word_encoder:
        wf = torch.zeros(world.vocab_size, cfg.word_feat_dim)
        for i in range(world.vocab_size):
            idxs = torch.randperm(cfg.word_feat_dim, generator=g)[: cfg.word_feat_active]
            wf[i, idxs] = 1.0
        word_feats = F.normalize(wf, dim=-1).to(dev)
    model = _ToyEncoders(world.vocab_size, cfg.feat_dim, cfg.dim, word_feats).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    curve, gini_curve, ref_curve = [], [], []
    for step in range(cfg.steps):
        if corpus is None:
            trials = [world.sample_trial() for _ in range(cfg.batch)]
        else:
            trials = [corpus[i] for i in rng.integers(len(corpus), size=cfg.batch)]
        ws, os_ = zip(*trials)
        w_idx, mask_w = _pad(list(ws), dev)
        o_idx, mask_s = _pad(list(os_), dev)

        ex = torch.as_tensor(rng.integers(n_tr, size=o_idx.shape), device=dev)
        feats = bank[o_idx, ex]
        feats = F.normalize(feats + cfg.feat_noise * torch.randn_like(feats), dim=-1)

        loss, stats = gavagai_loss(
            model.words(w_idx), model.slots(feats), mask_w, mask_s,
            kappa=cfg.kappa, eps=cfg.eps, rho=cfg.rho, tau=cfg.tau,
            null_prior=cfg.null_prior, use_null=cfg.use_null,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % cfg.eval_every == 0 or step == cfg.steps - 1:
            curve.append((step + 1, _embed_accuracy(model, bank, world)))
            gini_curve.append((step + 1, float(stats["slot_usage_gini"])))
            ref_curve.append((step + 1, float(stats["referential_mass"])))

    return {
        "accuracy": _embed_accuracy(model, bank, world),
        "hub_rate": _hub_rate(model, bank, world),
        "curve": curve,
        "gini_curve": gini_curve,
        "ref_curve": ref_curve,
        "model": model,
        "bank": bank,
    }


def _held_out(bank, world):
    """Mean embedding target per object, built only from held-out exemplars."""
    return bank[:, world.cfg.n_exemplars :]


@torch.no_grad()
def _embed_accuracy(model, bank, world) -> float:
    """Picture-vocabulary readout on *held-out* exemplars of each category."""
    ho = _held_out(bank, world)
    obj = F.normalize(model.slots(ho.reshape(-1, ho.shape[-1])).reshape(*ho.shape[:2], -1).mean(1), dim=-1)
    words = model.words(torch.arange(world.cfg.n_words, device=bank.device))
    pred = (words @ obj.t()).argmax(1).cpu().numpy()
    return float((pred == world.truth).mean())


@torch.no_grad()
def _hub_rate(model, bank, world) -> float:
    """Fraction of words whose nearest referent is a never-named background object.

    This is the direct measure of hub collapse.
    """
    if world.cfg.n_background == 0:
        return 0.0
    ho = _held_out(bank, world)
    obj = F.normalize(model.slots(ho.reshape(-1, ho.shape[-1])).reshape(*ho.shape[:2], -1).mean(1), dim=-1)
    words = model.words(torch.arange(world.cfg.n_words, device=bank.device))
    pred = (words @ obj.t()).argmax(1).cpu().numpy()
    return float((pred >= world.cfg.n_words).mean())


# ---------------------------------------------------------------------------
# Autoregressive-captioning simulation: the baseline that actually matches
# BabyVLM-V2.
#
# Every training stage of the published "baby model" is next-token prediction.
# There is no contrastive term anywhere, hence no batch-level marginal
# correction, and -- more importantly -- no latent word-to-region variable at
# all.  A captioning model can drive utterance perplexity down using global
# gist plus language priors without ever binding a word to a referent, which is
# our explanation for Picture Vocabulary sitting at chance.
#
# The proxy below keeps only what matters: the utterance is predicted from a
# *pooled* image representation, and the rows of the vocabulary head serve as
# word embeddings.  Those same rows provide the picture-vocabulary readout, so
# the baseline and our method are scored on identical parameters and differ by
# exactly one added loss term.
# ---------------------------------------------------------------------------


@dataclass
class ARConfig:
    dim: int = 64
    feat_dim: int = 128
    feat_noise: float = 0.35
    steps: int = 600
    batch: int = 64
    lr: float = 3e-3
    seed: int = 0
    eval_every: int = 100

    aux_weight: float = 0.0
    """0.0 = pure autoregressive captioning baseline.  > 0 adds the GAVAGAI
    referential-alignment term on top of the identical captioning loss."""

    tau: float = 0.07
    eps: float = 0.05
    rho: float | None = 1.0
    kappa: float = 0.0
    null_prior: float = 0.5
    use_null: bool = True


class _CaptionModel(nn.Module):
    def __init__(self, vocab: int, feat_dim: int, dim: int):
        super().__init__()
        self.vis = nn.Sequential(nn.Linear(feat_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.head = nn.Linear(dim, vocab, bias=False)
        nn.init.normal_(self.head.weight, std=0.02)

    def slots(self, feats):
        return F.normalize(self.vis(feats), dim=-1)

    def word_embeddings(self, idx):
        return F.normalize(self.head.weight[idx], dim=-1)


def run_ar_learner(world: ReferentialWorld, cfg: ARConfig, corpus: list | None = None) -> dict:
    """Train a captioning proxy, optionally with the GAVAGAI auxiliary loss."""
    torch.manual_seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(cfg.seed + 1)
    rng = np.random.default_rng(cfg.seed + 2)

    bank = world.appearance_bank(cfg.feat_dim, generator=g).to(dev)
    n_tr = world.cfg.n_exemplars
    model = _CaptionModel(world.vocab_size, cfg.feat_dim, cfg.dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    curve = []
    for step in range(cfg.steps):
        if corpus is None:
            trials = [world.sample_trial() for _ in range(cfg.batch)]
        else:
            trials = [corpus[i] for i in rng.integers(len(corpus), size=cfg.batch)]
        ws, os_ = zip(*trials)
        w_idx, mask_w = _pad(list(ws), dev)
        o_idx, mask_s = _pad(list(os_), dev)

        ex = torch.as_tensor(rng.integers(n_tr, size=o_idx.shape), device=dev)
        feats = bank[o_idx, ex]
        feats = F.normalize(feats + cfg.feat_noise * torch.randn_like(feats), dim=-1)
        slots = model.slots(feats)

        # Captioning loss: predict every uttered word from the pooled scene.
        pooled = (slots * mask_s.unsqueeze(-1)).sum(1) / mask_s.sum(1, keepdim=True).clamp_min(1)
        logits = model.head(F.normalize(pooled, dim=-1)) / cfg.tau
        lp = F.log_softmax(logits, dim=-1)
        picked = lp.gather(1, w_idx)
        loss = -(picked * mask_w).sum() / mask_w.sum().clamp_min(1)

        if cfg.aux_weight > 0:
            aux, _ = gavagai_loss(
                model.word_embeddings(w_idx), slots, mask_w, mask_s,
                kappa=cfg.kappa, eps=cfg.eps, rho=cfg.rho, tau=cfg.tau,
                null_prior=cfg.null_prior, use_null=cfg.use_null,
            )
            loss = loss + cfg.aux_weight * aux

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % cfg.eval_every == 0 or step == cfg.steps - 1:
            curve.append((step + 1, _ar_accuracy(model, bank, world)))

    return {"accuracy": _ar_accuracy(model, bank, world), "curve": curve, "model": model}


@torch.no_grad()
def _ar_accuracy(model, bank, world) -> float:
    """Picture-vocabulary readout from the vocabulary head rows, held-out exemplars."""
    ho = _held_out(bank, world)
    obj = F.normalize(model.slots(ho.reshape(-1, ho.shape[-1])).reshape(*ho.shape[:2], -1).mean(1), dim=-1)
    words = model.word_embeddings(torch.arange(world.cfg.n_words, device=bank.device))
    pred = (words @ obj.t()).argmax(1).cpu().numpy()
    return float((pred == world.truth).mean())
