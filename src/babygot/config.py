"""Configuration for BabyGOT.

Every hyper-parameter that matters for the paper's experiments is gathered here,
with defaults that fit a single Kaggle T4 GPU (16 GB).  A ``tiny`` preset is
provided for quick CPU / CI smoke runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass
class VisionConfig:
    """From-scratch ViT producing *patch-level* features (needed for grounding)."""

    image_size: int = 96          # rendered scene resolution
    patch_size: int = 16          # -> (96/16)^2 = 36 patch tokens
    in_channels: int = 3
    width: int = 192              # token / hidden dim d
    depth: int = 6                # transformer layers
    heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_cls: bool = False         # grounding uses patch tokens; CLS optional
    checkpoint: bool = True       # gradient checkpointing (T4 memory)


@dataclass
class TextConfig:
    vocab_size: int = 1024
    width: int = 192
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    max_len: int = 64
    dropout: float = 0.0
    checkpoint: bool = True


@dataclass
class TransportConfig:
    """Entropic optimal-transport alignment (the core of BabyGOT)."""

    temperature: float = 0.07         # cosine-temperature tau for the cost matrix
    epsilon: float = 0.05             # entropic regularisation of the OT coupling
    sinkhorn_iters: int = 20          # log-domain Sinkhorn iterations
    tol: float = 1e-6
    image_marginal: str = "learned"   # 'uniform' | 'learned'  (learned = saliency)
    text_marginal: str = "uniform"    # 'uniform' | 'learned'
    cost: str = "cosine"              # 'cosine' (1 - cos/tau) for now


@dataclass
class FusionConfig:
    """Gated cross-attention fusion: how visual features reach the LM."""

    summary_grid: int = 3            # spatial summary: pool patches to a grid x grid
    gate: bool = True                # token-wise dynamic gate (Looking-to-Learn)
    gate_bias: float = 0.0
    fuse_every: int = 1              # inject cross-attention into first N LM layers
    width: int = 192


@dataclass
class LossConfig:
    lm_weight: float = 1.0
    ot_weight: float = 0.1           # lambda_OT: alignment cost <P*, C> (scale-balanced)
    gate_weight: float = 0.0         # lambda_gate: gate regulariser (0 = learn freely)
    infonce_weight: float = 0.0      # optional auxiliary global InfoNCE (ablation)
    gate_reg: str = "entropy"        # 'entropy' | 'l1'  (only if gate_weight > 0)


@dataclass
class TrainConfig:
    seed: int = 0
    batch_size: int = 32
    steps: int = 4000
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 300
    grad_clip: float = 1.0
    amp: bool = True                 # bf16/fp16 autocast (T4 friendly)
    grad_accum: int = 1
    log_every: int = 100
    eval_every: int = 500
    save_dir: str = "runs"
    eval_batch_size: int = 64
    freeze_vision: bool = False
    device: Optional[str] = None     # None = auto (cuda if available)

    # ---- stage 3: instruction tuning (BabyVLM-V2 §3.2) ---------------------
    sft_steps: int = 800
    sft_lr: float = 1e-4
    sft_data: int = 4000


@dataclass
class Config:
    vision: VisionConfig = field(default_factory=VisionConfig)
    text: TextConfig = field(default_factory=TextConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # --- data generator -----------------------------------------------------
    n_train_scenes: int = 4000
    n_eval_scenes: int = 400
    seed: int = 0

    # --- experiment / ablations ---------------------------------------------
    method: str = "babygot"          # babygot | global_clip | babyllava | no_gate | no_ot

    @property
    def d_model(self) -> int:
        return self.vision.width

    def to_small(self) -> "Config":
        """A mid-size configuration: a fast but meaningful run (~2.5M params)."""
        c = replace(
            self,
            vision=replace(self.vision, image_size=96, patch_size=16, width=128,
                           depth=4, heads=4, checkpoint=False),
            text=replace(self.text, width=128, depth=4, heads=4, max_len=32,
                         vocab_size=1024, checkpoint=False),
            fusion=replace(self.fusion, width=128, summary_grid=3),
            train=replace(self.train, batch_size=64, steps=1500, lr=3e-4,
                          warmup_steps=120, log_every=100, eval_every=300,
                          amp=True, eval_batch_size=64, sft_steps=600,
                          sft_data=4000),
            n_train_scenes=4000,
            n_eval_scenes=150,
        )
        return c

    def to_tiny(self) -> "Config":
        """A fast configuration for CPU / CI smoke tests."""
        c = replace(
            self,
            vision=replace(self.vision, image_size=48, patch_size=16, width=64,
                           depth=2, heads=2, checkpoint=False),
            text=replace(self.text, width=64, depth=2, heads=2, max_len=32,
                         vocab_size=512, checkpoint=False),
            fusion=replace(self.fusion, width=64, summary_grid=3),
            train=replace(self.train, batch_size=16, steps=120, lr=1e-3,
                          warmup_steps=20, log_every=20, eval_every=60,
                          amp=False, eval_batch_size=32, sft_steps=60,
                          sft_data=600),
            n_train_scenes=600,
            n_eval_scenes=120,
        )
        return c


def make_config(method: str = "babygot", tiny: bool = False,
                small: bool = False) -> Config:
    """Build a config for a method and (optionally) a smaller preset."""
    c = Config()
    c.method = method
    if method == "babyllava":
        # BabyLLaVA-style: a single global visual token + pure autoregression.
        c.loss.ot_weight = 0.0
        c.fusion.gate = False
    if method == "global_clip":
        # CLIP/CVCL-style global InfoNCE alignment + generative LM.
        c.loss.ot_weight = 0.0
        c.loss.infonce_weight = 1.0
        c.fusion.gate = False
    if method == "no_gate":
        c.fusion.gate = False
    if method == "no_ot":
        # replace fine-grained OT with a token-wise (FILIP-style) surrogate
        c.loss.ot_weight = 0.0
        c.loss.infonce_weight = 0.0
    if tiny:
        c = c.to_tiny()
    elif small:
        c = c.to_small()
    return c
