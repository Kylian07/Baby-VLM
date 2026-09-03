"""Entropic optimal-transport primitives for cross-situational word learning.

The central object is a *coupling* ``P`` between the content words of an
utterance and the object slots of the co-occurring visual scene, augmented with
one extra "null" column that absorbs non-referential words (most of what a
caregiver says does not refer to anything currently visible).

Two knobs generate the entire family of objectives studied in this repo:

``eps``
    Entropic regularisation, i.e. a temperature.  ``eps -> 0`` drives ``P``
    towards a partial permutation matrix (hard mutual exclusivity); large
    ``eps`` gives a diffuse, purely associative coupling.

``rho``
    Strength of the *column* (slot) marginal constraint, i.e. the
    mutual-exclusivity pressure.  ``rho = 0`` recovers the row-wise softmax used
    by InfoNCE/CLIP **exactly**; ``rho -> inf`` gives balanced (doubly
    stochastic) OT.  Sweeping ``rho`` is therefore a continuous ablation from
    the standard contrastive baseline to full mutual exclusivity.

Everything is computed in the log domain.  The plan is differentiable, but in
practice we detach it and rely on Danskin's theorem: the gradient of the
entropic-OT value with respect to the cost is the optimal plan itself.
"""

from __future__ import annotations

import torch

NEG_INF = -1e30


def sinkhorn_log(
    cost: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    eps: float,
    rho: float | None = None,
    n_iter: int = 50,
    tol: float = 1e-7,
    check_every: int = 10,
) -> torch.Tensor:
    """Semi-relaxed entropic OT in the log domain.

    Solves (row marginal hard, column marginal soft)::

        min_P  <P, C> + eps * KL(P || a (x) b) + rho * KL(P^T 1 || b)
        s.t.   P 1 = a

    Args:
        cost: ``(..., N, M)`` transport cost.
        log_a: ``(..., N)`` log row marginal.  Use ``NEG_INF`` to mask padded
            rows; the corresponding rows of the plan come back as zero mass.
        log_b: ``(..., M)`` log column marginal.
        eps: entropic regularisation, > 0.
        rho: column-constraint strength.  ``None`` means ``inf`` (balanced);
            ``0.0`` disables the column constraint and yields a row-wise
            softmax.
        n_iter: maximum Sinkhorn iterations.
        tol: early-stopping threshold on the max potential update.  Set to 0 to
            disable the check entirely.
        check_every: how often to test convergence.  Each test calls ``.item()``,
            which forces a host synchronisation and destroys GPU throughput if
            done every iteration, so it is done sparingly.

    Returns:
        ``log_P`` of shape ``(..., N, M)``.  ``exp(log_P).sum(-1) == exp(log_a)``
        up to solver tolerance.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")

    # Column damping factor rho / (rho + eps).  1.0 -> hard, 0.0 -> no constraint.
    if rho is None:
        damp = 1.0
    else:
        if rho < 0:
            raise ValueError("rho must be non-negative or None")
        damp = rho / (rho + eps)

    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)

    for it in range(n_iter):
        # Row update (hard):  f_i = eps*log a_i - eps*LSE_j((g_j - C_ij)/eps)
        f_new = eps * log_a - eps * torch.logsumexp(
            (g.unsqueeze(-2) - cost) / eps, dim=-1
        )
        f_new = torch.nan_to_num(f_new, nan=NEG_INF, neginf=NEG_INF)

        if damp == 0.0:
            g_new = torch.zeros_like(g)
        else:
            # Column update (soft): g_j = damp * (eps*log b_j - eps*LSE_i(...))
            g_new = damp * (
                eps * log_b
                - eps * torch.logsumexp((f_new.unsqueeze(-1) - cost) / eps, dim=-2)
            )
            g_new = torch.nan_to_num(g_new, nan=0.0, neginf=NEG_INF)

        should_check = tol > 0 and (it + 1) % check_every == 0
        if should_check:
            delta = torch.maximum(
                (f_new - f).abs().amax(), (g_new - g).abs().amax()
            ).item()
        f, g = f_new, g_new
        if damp == 0.0:
            break  # single closed-form step; iterating changes nothing
        if should_check and delta < tol:
            break

    log_p = (f.unsqueeze(-1) + g.unsqueeze(-2) - cost) / eps
    return torch.nan_to_num(log_p, nan=NEG_INF, neginf=NEG_INF)


def null_augmented_cost(
    sim: torch.Tensor,
    kappa: torch.Tensor | float,
    slot_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the ``(..., N, M+1)`` cost with a trailing null column.

    ``sim`` are word-slot similarities (higher = better match).  Cost is the
    negated similarity; the null column has constant cost ``-kappa``, so a word
    is routed to null whenever no slot beats the ``kappa`` threshold.
    """
    cost = -sim
    if slot_mask is not None:
        cost = cost.masked_fill(~slot_mask.unsqueeze(-2), -NEG_INF)
    if not torch.is_tensor(kappa):
        kappa = torch.as_tensor(kappa, dtype=sim.dtype, device=sim.device)
    null_col = (-kappa).expand(*sim.shape[:-1], 1)
    return torch.cat([cost, null_col], dim=-1)


def referential_plan(
    sim: torch.Tensor,
    kappa: torch.Tensor | float,
    eps: float,
    rho: float | None,
    word_mask: torch.Tensor | None = None,
    slot_mask: torch.Tensor | None = None,
    null_prior: float = 0.5,
    n_iter: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve one episode's word-referent assignment problem.

    Args:
        sim: ``(B, N, M)`` word-slot similarities.
        kappa: scalar null threshold (may be a learnable tensor).
        eps: entropic temperature.
        rho: mutual-exclusivity strength (``None`` = balanced, ``0`` = softmax).
        word_mask: ``(B, N)`` bool, True for real content words.
        slot_mask: ``(B, M)`` bool, True for real slots.
        null_prior: expected fraction of words that refer to nothing visible.
            Sets the column marginal mass reserved for the null bin.

    Returns:
        ``plan``: ``(B, N, M)`` mass on real slots, rows summing to at most
        ``1/N_real``; and ``referential``: ``(B, N)`` in ``[0, 1]``, the
        fraction of each word's mass that landed on a real slot.  Words the
        model judges non-referential get ``referential ~ 0`` and are thereby
        excluded from the loss.
    """
    b, n, m = sim.shape
    cost = null_augmented_cost(sim, kappa, slot_mask)

    if word_mask is None:
        word_mask = torch.ones(b, n, dtype=torch.bool, device=sim.device)
    n_real = word_mask.sum(-1, keepdim=True).clamp(min=1)
    log_a = torch.where(
        word_mask,
        -torch.log(n_real.to(sim.dtype)).expand(b, n),
        torch.full_like(sim[..., 0], NEG_INF),
    )

    if slot_mask is None:
        slot_mask = torch.ones(b, m, dtype=torch.bool, device=sim.device)
    m_real = slot_mask.sum(-1, keepdim=True).clamp(min=1)
    slot_mass = (1.0 - null_prior) / m_real.to(sim.dtype)
    b_slots = torch.where(slot_mask, slot_mass.expand(b, m), torch.zeros_like(sim[..., 0, :]))
    b_full = torch.cat([b_slots, torch.full((b, 1), null_prior, dtype=sim.dtype, device=sim.device)], dim=-1)
    log_b = torch.log(b_full.clamp_min(1e-30))

    log_p = sinkhorn_log(cost, log_a, log_b, eps=eps, rho=rho, n_iter=n_iter)
    p = log_p.exp()

    plan = p[..., :m]
    row_total = p.sum(-1).clamp_min(1e-30)
    referential = (plan.sum(-1) / row_total).clamp(0.0, 1.0)
    referential = referential * word_mask.to(sim.dtype)
    return plan, referential


def sinkhorn_scale(joint: torch.Tensor, n_iter: int = 200, tol: float = 1e-10) -> torch.Tensor:
    """Sinkhorn-Knopp scaling of a non-negative matrix to doubly stochastic.

    Returns the scaled matrix ``D_r J D_c``.  By the Sinkhorn-Knopp theorem the
    scaling is unique up to a scalar, which is what makes Proposition 1 in
    ``METHOD.md`` well posed: ``log`` of the scaled matrix equals the pointwise
    mutual information of ``joint`` up to per-row and per-column constants.
    """
    m = joint.clone().double()
    r = torch.ones(m.shape[0], dtype=torch.float64)
    c = torch.ones(m.shape[1], dtype=torch.float64)
    for _ in range(n_iter):
        r_new = 1.0 / (m @ c).clamp_min(1e-300)
        c_new = 1.0 / (m.t() @ r_new).clamp_min(1e-300)
        if (c_new - c).abs().max() < tol and (r_new - r).abs().max() < tol:
            r, c = r_new, c_new
            break
        r, c = r_new, c_new
    return (r.unsqueeze(1) * m * c.unsqueeze(0)).to(joint.dtype)
