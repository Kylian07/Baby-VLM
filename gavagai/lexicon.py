"""Persistent cross-situational lexicon.

Solving optimal transport inside each batch only resolves ambiguity that is
resolvable *within that scene*.  Cross-situational learning is the claim that
ambiguity which is irresolvable in any single scene becomes resolvable once
evidence is aggregated across many scenes.

This module implements that aggregation.  Visual slots are soft-assigned to a
learned codebook of ``K`` concept prototypes; a running count matrix
``L in R^{V x K}`` accumulates how much alignment mass each word has received
for each concept; and the *pointwise mutual information* of that matrix is fed
back into the cost of the next episode's transport problem.

PMI rather than raw counts is essential and is the whole content of
Proposition 1 in ``METHOD.md``: raw counts rank referents by ``p(k | w)``, which
is dominated by the concept prior ``pi(k)``, whereas PMI ranks by
``p(w, k) / (p(w) pi(k))``, whose argmax is ``argmax_k p(w | k)`` and is
therefore invariant to how often a concept happens to be on screen.

The prototype codebook is kept from collapsing by assigning slots with a
Sinkhorn-balanced (SwAV-style) step rather than a plain softmax -- the same
mathematical device as the main objective, applied one level down.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ot import NEG_INF, sinkhorn_log


class CrossSituationalLexicon(nn.Module):
    """Word x concept evidence accumulated across episodes.

    All state is held in buffers and updated under ``no_grad``: the lexicon is a
    *statistic* of the data, not a parameter to be optimised, which keeps the
    E-step consistent with the stochastic-EM derivation.
    """

    def __init__(
        self,
        vocab_size: int,
        n_prototypes: int = 512,
        dim: int = 256,
        gamma: float = 0.999,
        proto_momentum: float = 0.99,
        smoothing: float = 1e-2,
        assign_temp: float = 0.1,
        bonus_scale: float = 0.3,
        max_pmi: float = 4.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.proto_momentum = proto_momentum
        self.smoothing = smoothing
        self.assign_temp = assign_temp
        self.bonus_scale = bonus_scale
        self.max_pmi = max_pmi

        self.register_buffer("prototypes", F.normalize(torch.randn(n_prototypes, dim), dim=-1))
        self.register_buffer("counts", torch.zeros(vocab_size, n_prototypes))
        self.register_buffer("n_updates", torch.zeros((), dtype=torch.long))

    # -- assignment ---------------------------------------------------------

    @torch.no_grad()
    def assign(self, slot_emb: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        """Soft-assign slots to prototypes with a batch-balanced Sinkhorn step.

        Args:
            slot_emb: ``(B, M, D)`` L2-normalised slot embeddings.
            slot_mask: ``(B, M)`` bool.

        Returns:
            ``(B, M, K)`` assignment simplex.
        """
        b, m, _ = slot_emb.shape
        flat = slot_emb.reshape(b * m, -1)
        mask = slot_mask.reshape(b * m)
        scores = flat @ self.prototypes.t()

        n_real = mask.sum().clamp_min(1)
        log_a = torch.where(
            mask,
            torch.full_like(scores[:, 0], -torch.log(n_real.float())),
            torch.full_like(scores[:, 0], NEG_INF),
        )
        k = self.prototypes.shape[0]
        log_b = torch.full((k,), -torch.log(torch.tensor(float(k))), device=scores.device)

        log_q = sinkhorn_log(
            (-scores).unsqueeze(0),
            log_a.unsqueeze(0),
            log_b.unsqueeze(0),
            eps=self.assign_temp,
            rho=None,
            n_iter=3,
        )[0]
        q = log_q.exp()
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
        return q.reshape(b, m, k)

    # -- read ---------------------------------------------------------------

    @torch.no_grad()
    def pmi(self) -> torch.Tensor:
        """Smoothed, clipped PMI of the accumulated word x concept counts."""
        c = self.counts + self.smoothing
        total = c.sum()
        pmi = torch.log(c * total / (c.sum(1, keepdim=True) * c.sum(0, keepdim=True)))
        return pmi.clamp(-self.max_pmi, self.max_pmi)

    @torch.no_grad()
    def bonus(
        self, word_ids: torch.Tensor, q: torch.Tensor, warmup: int = 200
    ) -> torch.Tensor | None:
        """Expected accumulated PMI for every (word, slot) pair.

        Returns ``None`` during warmup, when the lexicon carries no signal yet
        and would only inject noise into the transport cost.
        """
        if int(self.n_updates) < warmup:
            return None
        rows = self.pmi()[word_ids]                      # (B, N, K)
        bonus = torch.einsum("bnk,bmk->bnm", rows, q)    # (B, N, M)
        return self.bonus_scale * bonus / self.max_pmi

    # -- write --------------------------------------------------------------

    @torch.no_grad()
    def update(
        self,
        word_ids: torch.Tensor,
        q: torch.Tensor,
        plan: torch.Tensor,
        word_mask: torch.Tensor,
        slot_emb: torch.Tensor | None = None,
    ) -> None:
        """Accumulate one batch of alignments into the lexicon and codebook.

        Args:
            word_ids: ``(B, N)`` vocabulary indices.
            q: ``(B, M, K)`` slot-to-prototype assignments.
            plan: ``(B, N, M)`` transport mass on real slots.
            word_mask: ``(B, N)`` bool.
            slot_emb: ``(B, M, D)``; if given, prototypes are EMA-updated.
        """
        contrib = torch.einsum("bnm,bmk->bnk", plan, q)          # (B, N, K)
        contrib = contrib * word_mask.unsqueeze(-1).to(contrib.dtype)

        flat_ids = word_ids.reshape(-1)
        flat_contrib = contrib.reshape(-1, contrib.shape[-1])
        if self.gamma < 1.0:
            self.counts.mul_(self.gamma)
        self.counts.index_add_(0, flat_ids, flat_contrib.to(self.counts.dtype))

        if slot_emb is not None:
            b, m, d = slot_emb.shape
            flat_slots = slot_emb.reshape(b * m, d)
            flat_q = q.reshape(b * m, -1)
            num = flat_q.t() @ flat_slots                        # (K, D)
            den = flat_q.sum(0).unsqueeze(-1).clamp_min(1e-6)
            target = F.normalize(num / den, dim=-1)
            used = (flat_q.sum(0) > 1e-4).unsqueeze(-1)
            new = F.normalize(
                self.proto_momentum * self.prototypes + (1 - self.proto_momentum) * target,
                dim=-1,
            )
            self.prototypes.copy_(torch.where(used, new, self.prototypes))

        self.n_updates += 1

    @torch.no_grad()
    def known_words(self, threshold: float = 1.0, min_count: float = 5.0) -> torch.Tensor:
        """Boolean mask of words with a committed referent.

        A word counts as "known" once some concept exceeds ``threshold`` PMI and
        the word has been heard enough times.  Tracking this quantity over
        training is what produces the vocabulary-growth curve.
        """
        enough = self.counts.sum(1) >= min_count
        peaked = self.pmi().max(1).values >= threshold
        return enough & peaked
