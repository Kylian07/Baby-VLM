"""The GAVAGAI training objective.

One episode = one visual scene (a set of ``M`` object slots) plus one utterance
(a set of ``N`` content words).  Which word goes with which slot is latent.

The loss factorises into two steps, in the spirit of stochastic EM:

**E-step** (no gradient) -- solve a semi-relaxed entropic OT problem to decide
*what* is aligned to *what*, and how much of each word is non-referential.

**M-step** (gradient) -- a soft-target InfoNCE over every slot in the batch,
using the E-step plan as the target distribution and the per-word referential
mass as a sample weight.

Setting ``rho = 0`` makes the E-step a plain row-wise softmax, which recovers
standard region-word contrastive learning exactly.  Setting ``use_null=False``
additionally removes the non-referential escape hatch.  The full method is
``rho > 0`` with the null bin enabled.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .ot import referential_plan


def _flatten_slots(slot_emb: torch.Tensor, slot_mask: torch.Tensor):
    b, m, d = slot_emb.shape
    return slot_emb.reshape(b * m, d), slot_mask.reshape(b * m)


def gavagai_loss(
    word_emb: torch.Tensor,
    slot_emb: torch.Tensor,
    word_mask: torch.Tensor,
    slot_mask: torch.Tensor,
    *,
    kappa: torch.Tensor | float = 0.0,
    eps: float = 0.05,
    rho: float | None = 1.0,
    tau: float = 0.07,
    null_prior: float = 0.5,
    use_null: bool = True,
    lexicon_bonus: torch.Tensor | None = None,
    ot_iters: int = 30,
    symmetric: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Compute the objective for a batch of episodes.

    Args:
        word_emb: ``(B, N, D)`` L2-normalised content-word embeddings.
        slot_emb: ``(B, M, D)`` L2-normalised visual slot embeddings.
        word_mask: ``(B, N)`` bool.
        slot_mask: ``(B, M)`` bool.
        kappa: null-bin threshold on the cosine scale (may be learnable).
        eps: entropic temperature of the E-step.
        rho: mutual-exclusivity strength.  ``0`` -> row softmax, ``None`` -> balanced.
        tau: temperature of the M-step contrastive term.
        null_prior: column mass reserved for the null bin.
        use_null: if False, every word is forced onto some slot.
        lexicon_bonus: ``(B, N, M)`` accumulated cross-situational PMI evidence,
            added to the similarity before the E-step.  Detached by the caller.
        symmetric: also apply the slot -> word direction.

    Returns:
        ``(loss, stats)``.
    """
    b, n, _ = word_emb.shape
    m = slot_emb.shape[1]

    sim_in = torch.einsum("bnd,bmd->bnm", word_emb, slot_emb)

    with torch.no_grad():
        e_sim = sim_in.detach()
        if lexicon_bonus is not None:
            e_sim = e_sim + lexicon_bonus
        plan, referential = referential_plan(
            e_sim,
            kappa=kappa if use_null else -1e4,
            eps=eps,
            rho=rho,
            word_mask=word_mask,
            slot_mask=slot_mask,
            null_prior=null_prior if use_null else 0.0,
            n_iter=ot_iters,
        )
        if not use_null:
            referential = word_mask.to(sim_in.dtype)
        target = plan / plan.sum(-1, keepdim=True).clamp_min(1e-12)
        target = target * slot_mask.unsqueeze(-2).to(target.dtype)

    flat_slots, flat_mask = _flatten_slots(slot_emb, slot_mask)
    logits = torch.einsum("bnd,kd->bnk", word_emb, flat_slots) / tau
    neg = torch.finfo(logits.dtype).min / 4
    logits = logits.masked_fill(~flat_mask.view(1, 1, -1), neg)
    log_prob = F.log_softmax(logits, dim=-1)

    # Scatter the within-episode target into the batch-wide slot axis.
    full_target = torch.zeros_like(log_prob)
    idx = (torch.arange(b, device=word_emb.device) * m).view(b, 1, 1) + torch.arange(
        m, device=word_emb.device
    ).view(1, 1, m)
    full_target.scatter_(2, idx.expand(b, n, m), target)

    w = referential * word_mask.to(referential.dtype)
    denom = w.sum().clamp_min(1e-6)
    xent = torch.where(full_target > 0, full_target * log_prob, torch.zeros_like(log_prob))
    loss_w2s = -xent.sum(-1).mul(w).sum() / denom

    stats = {
        "referential_mass": (w.sum() / word_mask.sum().clamp_min(1)).detach(),
        "plan_entropy": _plan_entropy(target, word_mask).detach(),
        "slot_usage_gini": _slot_gini(plan, slot_mask).detach(),
        # The plan is returned so the caller can accumulate it into the
        # cross-situational lexicon; it is already detached (E-step).
        "plan": plan,
        "referential": referential,
    }

    if not symmetric:
        return loss_w2s, stats

    flat_words = word_emb.reshape(b * n, -1)
    flat_wmask = word_mask.reshape(b * n)
    s_logits = torch.einsum("bmd,kd->bmk", slot_emb, flat_words) / tau
    s_logits = s_logits.masked_fill(~flat_wmask.view(1, 1, -1), neg)
    s_log_prob = F.log_softmax(s_logits, dim=-1)

    s_target_local = plan.transpose(1, 2)  # (B, M, N)
    s_target_local = s_target_local / s_target_local.sum(-1, keepdim=True).clamp_min(1e-12)
    s_target_local = s_target_local * word_mask.unsqueeze(-2).to(s_target_local.dtype)
    s_full = torch.zeros_like(s_log_prob)
    widx = (torch.arange(b, device=word_emb.device) * n).view(b, 1, 1) + torch.arange(
        n, device=word_emb.device
    ).view(1, 1, n)
    s_full.scatter_(2, widx.expand(b, m, n), s_target_local)

    slot_w = plan.sum(1) * slot_mask.to(plan.dtype)  # how referential each slot is
    s_denom = slot_w.sum().clamp_min(1e-6)
    s_xent = torch.where(s_full > 0, s_full * s_log_prob, torch.zeros_like(s_log_prob))
    loss_s2w = -s_xent.sum(-1).mul(slot_w).sum() / s_denom

    return 0.5 * (loss_w2s + loss_s2w), stats


def _plan_entropy(target: torch.Tensor, word_mask: torch.Tensor) -> torch.Tensor:
    p = target.clamp_min(1e-12)
    ent = -(p * p.log()).sum(-1)
    return (ent * word_mask).sum() / word_mask.sum().clamp_min(1)


def _slot_gini(plan: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
    """How unevenly words are spread over slots -- 0 = uniform, 1 = single hub.

    This is the direct measurement of the hub collapse that the column marginal
    constraint is designed to make infeasible.
    """
    usage = (plan.sum(1) * slot_mask.to(plan.dtype))
    usage = usage / usage.sum(-1, keepdim=True).clamp_min(1e-12)
    k = slot_mask.sum(-1).clamp_min(1).to(usage.dtype)
    # normalised Herfindahl index, rescaled to [0, 1]
    hhi = (usage**2).sum(-1)
    return ((hhi - 1.0 / k) / (1.0 - 1.0 / k).clamp_min(1e-6)).clamp_min(0.0).mean()


def infonce_loss(
    text_emb: torch.Tensor, image_emb: torch.Tensor, tau: float = 0.07
) -> torch.Tensor:
    """Standard global (utterance <-> image) CLIP loss, used as the base term."""
    logits = text_emb @ image_emb.t() / tau
    labels = torch.arange(text_emb.shape[0], device=text_emb.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
