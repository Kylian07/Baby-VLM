"""BabyGOT: Grounded Optimal-Transport pretraining model.

Architecture
------------
  image  ->  ViT (from scratch)  ->  patch features z_v in R^{N x d}
  text   ->  word embeddings       ->  z_t in R^{M x d}  (context-free, for OT)
  fusion ->  spatial summary tokens G = AdaptivePool(z_v)  (grid, row-major)
             h_t <- h_t + gamma_t * CrossAttn(h_t, G)      (token-wise gate)
  LM     ->  causal TinyGPT on the fused tokens           (generative arm)

Objective (hybrid, see objectives.py / paper)
  L = L_LM + lambda_OT * <P*, C> + lambda_gate * R(gamma) [+ aux InfoNCE]

The baselines ('babyllava', 'global_clip') replace the gated fusion with a single
global visual token prepended to the LM (the LLaVA/BabyVLM connector).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Config
from .fusion import GatedFusion
from .text import TinyGPT
from .transport import TransportAlign, infonce, tokenwise_max_alignment
from .vision import ViT


class GlobalProjector(nn.Module):
    """BabyLLaVA-style MLP connector: pooled patches -> 1 global token."""

    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, z_v: torch.Tensor) -> torch.Tensor:
        return self.mlp(z_v.mean(dim=1)).unsqueeze(1)     # (B, 1, d)


class BabyGOT(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        assert cfg.text.width == d, "vision and text widths must match"

        self.vision = ViT(cfg.vision)
        self.text = TinyGPT(cfg.text)
        self.gated = cfg.method not in ("babyllava", "global_clip")
        if self.gated:
            self.fusion = GatedFusion(cfg.fusion, d)
        else:
            self.projector = GlobalProjector(d)
        if cfg.loss.ot_weight > 0:
            self.align = TransportAlign(cfg.transport, d)

    # ------------------------------------------------------------------ #
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.vision(image)                            # (B, N, d)

    @staticmethod
    def _lm_loss(logits: torch.Tensor, ids: torch.Tensor, offset: int):
        """Next-token loss; first ``offset`` positions are visual tokens."""
        shift_logits = logits[:, offset:-1].contiguous()
        shift_ids = ids[:, 1:].contiguous()
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_ids.view(-1))

    def forward(self, image: torch.Tensor, ids: torch.Tensor,
                align: bool = True):
        """Returns a dict: {loss, lm_loss, [ot_cost, P, gamma, ...]}.

        ``align=False`` disables the alignment / gate-regulariser terms and
        returns the pure autoregressive loss (used by the instruction-tuning
        stage, mirroring LLaVA / BabyVLM-V2 stage 3).
        """
        cfg = self.cfg
        z_v = self.encode_image(image)                      # (B, N, d)
        tok = self.text.tok_emb(ids)                        # (B, M, d)
        out: dict = {}

        # ---- fusion ------------------------------------------------------ #
        if self.gated:
            h_fused, gamma = self.fusion(tok, z_v)
            hidden = self.text.forward_emb(h_fused)
            offset = 0
            out["gamma"] = gamma
        else:
            g = self.projector(z_v)
            hidden = self.text(ids, features=g)
            offset = 1
        logits = self.text.logits(hidden)
        lm_loss = self._lm_loss(logits, ids, offset)
        out["lm_loss"] = lm_loss

        # ---- hybrid objective -------------------------------------------- #
        z_t = tok                                          # context-free words
        loss = cfg.loss.lm_weight * lm_loss

        if align:
            if cfg.loss.ot_weight > 0:
                a = self.align(z_v, z_t)
                out["ot_cost"] = a["cost"]
                out["P"] = a["P"]
                loss = loss + cfg.loss.ot_weight * a["cost"]

            if cfg.method == "no_ot":
                l = tokenwise_max_alignment(z_v, z_t, cfg.transport.temperature)
                out["filip_loss"] = l
                loss = loss + 1.0 * l

            if cfg.loss.infonce_weight > 0:
                l = infonce(z_v, z_t, cfg.transport.temperature)
                out["infonce"] = l
                loss = loss + cfg.loss.infonce_weight * l

            if self.gated and cfg.fusion.gate and cfg.loss.gate_weight > 0:
                reg = gate_regularizer(gamma, cfg.loss.gate_reg)
                out["gate_reg"] = reg
                loss = loss + cfg.loss.gate_weight * reg

        out["loss"] = loss
        return out

    # ------------------------------------------------------------------ #
    def seq_logp(self, image: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Sum of log p(x_t | x_<t, image) over a sequence, per sample. (B,).

        This is the *only* quantity the discriminative probes need: choosing the
        option / image that maximises this likelihood is exact Bayesian decoding
        under the model's autoregressive distribution.
        """
        z_v = self.encode_image(image)
        tok = self.text.tok_emb(ids)
        if self.gated:
            h_fused, _ = self.fusion(tok, z_v)
            hidden = self.text.forward_emb(h_fused)
            offset = 0
        else:
            g = self.projector(z_v)
            hidden = self.text(ids, features=g)
            offset = 1
        logits = self.text.logits(hidden)
        shift_logits = logits[:, offset:-1].contiguous()
        shift_ids = ids[:, 1:].contiguous()
        per_tok = -nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_ids.view(-1), reduction="none")
        B, T = shift_ids.shape
        return per_tok.view(B, T).sum(dim=1)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, image: torch.Tensor, ids: torch.Tensor,
                 max_new: int = 16, temperature: float = 1.0,
                 top_k: int = 0, eos_id: int | None = None) -> torch.Tensor:
        """Autoregressive decoding (used by captioning / QA probes)."""
        B = ids.shape[0]
        z_v = self.encode_image(image)
        tok = self.text.tok_emb(ids)
        if self.gated:
            h_fused, _ = self.fusion(tok, z_v)
            x = h_fused
        else:
            g = self.projector(z_v)
            x = torch.cat([g, tok], dim=1)                 # (B, 1+M, d)

        T = x.shape[1]
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + self.text.pos(pos[:, :T])
        generated = []
        for _ in range(max_new):
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            h = x
            for blk in self.text.blocks:
                h = blk(h, mask)
            logits = self.text.head(self.text.ln(h))[:, -1] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)              # (B, 1)
            generated.append(nxt)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
            emb = self.text.tok(nxt) + self.text.pos(
                torch.tensor([[T]], device=x.device).expand(B, -1))
            x = torch.cat([x, emb], dim=1)
            T += 1
        return torch.cat(generated, dim=1) if generated else torch.zeros(
            B, 0, dtype=torch.long, device=ids.device)


def gate_regularizer(gamma: torch.Tensor, kind: str = "entropy") -> torch.Tensor:
    """Keep the gate informative: entropy discourages collapse to 0/1; L1 pushes
    sparsity.  Both yield an interpretable content/function-word split."""
    g = gamma.clamp(1e-6, 1 - 1e-6)
    if kind == "entropy":
        return -(g * torch.log(g) + (1 - g) * torch.log(1 - g)).mean()
    return g.mean()  # L1 (gamma in (0,1))
