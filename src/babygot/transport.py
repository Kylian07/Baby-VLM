"""Entropic optimal transport between image patches and words.

This module implements the mathematical core of BabyGOT.

Notation
--------
  z_v in R^{N x d} : patch-level visual features   (N = number of patches)
  z_t in R^{M x d} : token-level linguistic features (M = number of words)

  S_ij  = <z_v^i, z_t^j> / tau            (cosine similarity, temperature tau)
  C_ij  = 1 - S_ij                        (alignment cost; lower = more similar)

  The *referential alignment* between words and image regions is modelled as a
  coupling  P in U(a, b) = {P >= 0 : P 1 = a, P^T 1 = b}  (a, b marginals).
  Following Cuturi (2013) we solve the entropy-regularised problem

      P* = argmin_{P in U(a,b)}  <P, C> - eps * H(P)          (H = Shannon entropy)

  with log-domain Sinkhorn iterations.  P*_ij is the (soft) probability that word
  j "refers to" region i — i.e. a differentiable, many-to-many generalisation of
  the hard word<->object mappings infants infer by cross-situational learning.

  Relations to baselines (proved in the paper):
    * eps -> 0      : P* converges to the min-cost matching (FILIP-style 1-1 map).
    * N = M = 1     : <P*, C> reduces to the standard InfoNCE / CLIP contrastive
                      objective on global embeddings.
    * eps -> +inf   : P* -> a b^T (uniform, no alignment signal).
  BabyGOT therefore *strictly generalises* the global CLIP/CVCL and BabyLLaVA
  alignment losses used by the BabyVLM baselines.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import TransportConfig


def cosine_cost(z_v: torch.Tensor, z_t: torch.Tensor, tau: float) -> torch.Tensor:
    """C_ij = 1 - cos(z_v^i, z_t^j) / tau.   (B, N, M)."""
    z_v = nn.functional.normalize(z_v, dim=-1)
    z_t = nn.functional.normalize(z_t, dim=-1)
    sim = torch.einsum("bnd,bmd->bnm", z_v, z_t) / tau
    return 1.0 - sim


@torch.no_grad()
def _uniform_marginal(z: torch.Tensor) -> torch.Tensor:
    B, N, _ = z.shape
    return torch.full((B, N), 1.0 / N, device=z.device, dtype=z.dtype)


def sinkhorn(C: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
             eps: float, iters: int = 20, tol: float = 1e-6) -> torch.Tensor:
    """Log-domain Sinkhorn.  Returns coupling P in R^{B x N x M}."""
    B, N, M = C.shape
    loga, logb = torch.log(a + 1e-12), torch.log(b + 1e-12)
    u = torch.zeros(B, N, device=C.device, dtype=C.dtype)
    v = torch.zeros(B, M, device=C.device, dtype=C.dtype)
    inv = 1.0 / eps
    for _ in range(iters):
        u_new = eps * (loga - torch.logsumexp((v.unsqueeze(1) - C) * inv, dim=-1))
        v_new = eps * (logb - torch.logsumexp((u_new.unsqueeze(-1) - C) * inv, dim=-2))
        if tol > 0 and torch.max(torch.abs(u_new - u)) < tol and \
                torch.max(torch.abs(v_new - v)) < tol:
            u, v = u_new, v_new
            break
        u, v = u_new, v_new
    P = torch.exp((u.unsqueeze(-1) + v.unsqueeze(1) - C) * inv)
    return P


class MarginalHead(nn.Module):
    """Learnable referential saliency over patches / words (optional marginals)."""

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.proj(z).squeeze(-1)          # (B, N)
        return torch.softmax(logits, dim=-1)


class TransportAlign(nn.Module):
    """The OT alignment head.

    forward(z_v, z_t) -> dict(P, cost, S) with
        P    : the referential coupling (B, N, M)
        cost : <P*, C>  (mean over the batch)
    """

    def __init__(self, cfg: TransportConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.vis_marg = MarginalHead(d_model) if cfg.image_marginal == "learned" else None
        self.txt_marg = MarginalHead(d_model) if cfg.text_marginal == "learned" else None

    def _marginals(self, z_v: torch.Tensor, z_t: torch.Tensor):
        a = self.vis_marg(z_v) if self.vis_marg is not None else _uniform_marginal(z_v)
        b = self.txt_marg(z_t) if self.txt_marg is not None else _uniform_marginal(z_t)
        return a, b

    def forward(self, z_v: torch.Tensor, z_t: torch.Tensor):
        C = cosine_cost(z_v, z_t, self.cfg.temperature)
        a, b = self._marginals(z_v, z_t)
        P = sinkhorn(C, a, b, self.cfg.epsilon, self.cfg.sinkhorn_iters, self.cfg.tol)
        cost = (P * C).sum(dim=(1, 2)).mean()
        return {"P": P, "cost": cost, "C": C}


def tokenwise_max_alignment(z_v: torch.Tensor, z_t: torch.Tensor,
                            tau: float) -> torch.Tensor:
    """FILIP-style fine-grained surrogate (ablation): max-similarity per word.

    L = - (1/M) sum_j max_i sim_ij.  Not a metric, no soft marginals — used to
    isolate the contribution of the OT coupling itself.
    """
    z_v = nn.functional.normalize(z_v, dim=-1)
    z_t = nn.functional.normalize(z_t, dim=-1)
    sim = torch.einsum("bnd,bmd->bnm", z_v, z_t) / tau
    return -sim.max(dim=1).values.mean()


def infonce(z_v: torch.Tensor, z_t: torch.Tensor, tau: float) -> torch.Tensor:
    """Global bidirectional InfoNCE (the CLIP/CVCL baseline loss)."""
    v = nn.functional.normalize(z_v.mean(dim=1), dim=-1)
    t = nn.functional.normalize(z_t.mean(dim=1), dim=-1)
    logits = v @ t.T / tau
    labels = torch.arange(v.shape[0], device=v.device)
    loss = (nn.functional.cross_entropy(logits, labels)
            + nn.functional.cross_entropy(logits.T, labels)) / 2
    return loss
