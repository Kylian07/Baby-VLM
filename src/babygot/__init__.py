"""BabyGOT — Grounded Optimal-Transport pretraining for sample-efficient VLMs.

A research codebase for the BabyVLM workshop (NeurIPS 2026).  See the
accompanying paper (``paper/paper.md``) for the mathematics; the implementation
is intentionally dependency-light (PyTorch + NumPy only).
"""

from . import (analysis, benchmarks, config, data, evaluate, fusion, model,
               text, tokenizer)
from . import train, transport, vision

__version__ = "0.1.0"

__all__ = [
    "analysis", "benchmarks", "config", "data", "evaluate", "fusion", "model",
    "text", "tokenizer", "train", "transport", "vision", "__version__",
]
