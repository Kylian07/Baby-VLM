"""Fast correctness tests (CPU, tiny config).  Run:  PYTHONPATH=src pytest -q"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from babygot.config import make_config
from babygot.data import SceneGenerator, render_scene
from babygot.model import BabyGOT
from babygot.tokenizer import Tokenizer
from babygot.transport import cosine_cost, sinkhorn, _uniform_marginal
from babygot.benchmarks import build_suite
from babygot.evaluate import evaluate_suite


def _tiny():
    cfg = make_config("babygot", tiny=True)
    tok = Tokenizer.build(["the red ball is on the left", "a blue block"],
                          max_size=cfg.text.vocab_size)
    return cfg, tok


def test_sinkhorn_marginals():
    torch.manual_seed(0)
    B, N, M = 3, 9, 7
    C = torch.randn(B, N, M)
    a = _uniform_marginal(torch.zeros(B, N, 1))
    b = _uniform_marginal(torch.zeros(B, M, 1))
    P = sinkhorn(C, a, b, eps=0.1, iters=200)
    assert P.shape == (B, N, M)
    assert torch.all(P >= 0)
    assert torch.allclose(P.sum(-1), a, atol=1e-3)
    assert torch.allclose(P.sum(-2), b, atol=1e-3)


def test_sinkhorn_sharpness():
    """eps -> 0 concentrates mass on min-cost pairs; eps -> inf goes uniform."""
    torch.manual_seed(1)
    N, M = 4, 4
    C = torch.randn(1, N, M)
    a = _uniform_marginal(torch.zeros(1, N, 1))
    b = _uniform_marginal(torch.zeros(1, M, 1))
    sharp = sinkhorn(C, a, b, eps=0.01, iters=500)
    flat = sinkhorn(C, a, b, eps=50.0, iters=50)
    # With uniform marginals 1/N, a hard matching has entries exactly 1/N=0.25;
    # sharp should put ~all row mass on one entry (others ~0), flat is uniform.
    assert sharp.max() > 0.2 and sharp.min() < 1e-3
    assert torch.allclose(flat, flat.mean(), atol=1e-2)


def test_forward_generate():
    cfg, tok = _tiny()
    gen = SceneGenerator(seed=0, image_size=cfg.vision.image_size)
    s = gen.sample()
    model = BabyGOT(cfg, vocab_size=cfg.text.vocab_size)
    ids = torch.tensor([tok.encode(s["caption"])])
    out = model(s["image"].unsqueeze(0), ids)
    assert torch.isfinite(out["loss"])
    assert "P" in out and out["P"].shape[1] == model.encode_image(
        s["image"].unsqueeze(0)).shape[1]
    g = model.generate(s["image"].unsqueeze(0),
                       torch.tensor([[tok.bos_id()]]), max_new=6)
    assert g.shape[1] == 6


def test_scene_render():
    img = render_scene([{"shape": "circle", "color": "red", "size": "big",
                         "cell": (1, 1)}], image_size=48)
    assert img.shape == (3, 48, 48)
    assert img.min() >= 0 and img.max() <= 1


def test_benchmarks_and_eval():
    cfg, tok = _tiny()
    model = BabyGOT(cfg, vocab_size=cfg.text.vocab_size)
    suite = build_suite(n_each=4, image_size=cfg.vision.image_size, device="cpu")
    res = evaluate_suite(model, suite, tok)
    assert "overall_choice_acc" in res
    for name in ["picture_vocabulary", "counting", "localization_h",
                 "who_has_more", "spatial_details", "memory", "vtwt_phrase",
                 "vtwt_image"]:
        assert name in res and "acc" in res[name]


def test_all_methods_build():
    for m in ["babygot", "global_clip", "babyllava", "no_gate", "no_ot"]:
        cfg = make_config(m, tiny=True)
        tok = Tokenizer.build(["a red ball"], max_size=cfg.text.vocab_size)
        model = BabyGOT(cfg, vocab_size=cfg.text.vocab_size)
        out = model(torch.randn(1, 3, cfg.vision.image_size,
                                cfg.vision.image_size),
                    torch.tensor([tok.encode("a red ball")]))
        assert torch.isfinite(out["loss"])
