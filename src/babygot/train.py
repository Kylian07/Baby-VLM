"""Training loop for BabyGOT — single-GPU (Kaggle T4) friendly.

Features: mixed precision (bf16 on T4), gradient accumulation, warmup + cosine
schedule, gradient clipping, per-step logging and periodic evaluation on the
developmentally-aligned probe suite.
"""

from __future__ import annotations

import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .benchmarks import build_suite
from .config import Config
from .data import (InstructionGenerator, SceneDataset, SceneGenerator, collate,
                   qa_text)
from .evaluate import evaluate_suite, format_results
from .model import BabyGOT
from .tokenizer import Tokenizer


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tokenizer(cfg: Config, captions) -> Tokenizer:
    """Seed the vocabulary with every word the probes can ask about, so the
    benchmark is not confounded by out-of-vocabulary answers."""
    from .data import COLORS, NUMBERS, POS_LABEL, ROW_LABEL, SHAPE_NOUN, SHAPES, SIZES

    template_words = set()
    for s in SHAPES:
        template_words.add(SHAPE_NOUN[s])
        template_words.add(SHAPE_NOUN[s] + "s")
    template_words |= set(COLORS) | set(SIZES) | set(NUMBERS)
    template_words |= set(POS_LABEL) | set(ROW_LABEL)
    template_words |= {"a", "the", "is", "on", "in", "look", "at", "where",
                       "how", "many", "are", "there", "which", "side", "has",
                       "more", "yes", "no", "before", "was", "same", "two",
                       "objects", "answer", "with", "one", "word", "find"}
    texts = list(captions) + list(template_words)
    return Tokenizer.build(texts, min_freq=1, max_size=cfg.text.vocab_size)


def make_loader(cfg: Config, tokenizer: Tokenizer, split: str):
    n = cfg.n_train_scenes if split == "train" else cfg.n_eval_scenes
    gen = SceneGenerator(seed=cfg.seed + (0 if split == "train" else 1000),
                         image_size=cfg.vision.image_size, device="cpu")
    ds = SceneDataset(n, gen)
    shuffle = split == "train"
    return DataLoader(
        ds, batch_size=cfg.train.batch_size, shuffle=shuffle, drop_last=shuffle,
        collate_fn=lambda b: collate(b, tokenizer, cfg.text.max_len))


def get_device(cfg: Config) -> torch.device:
    if cfg.train.device:
        return torch.device(cfg.train.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_dtype(device: torch.device):
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def train(cfg: Config, save_dir: str = "runs"):
    seed_everything(cfg.train.seed)
    device = get_device(cfg)
    os.makedirs(save_dir, exist_ok=True)

    # ---- data + tokenizer -------------------------------------------------
    gen = SceneGenerator(seed=cfg.seed, image_size=cfg.vision.image_size,
                         device="cpu")
    train_captions = [gen.sample()["caption"] for _ in range(min(cfg.n_train_scenes, 2000))]
    tokenizer = build_tokenizer(cfg, train_captions)
    assert len(tokenizer) <= cfg.text.vocab_size

    train_loader = make_loader(cfg, tokenizer, "train")

    # ---- model ------------------------------------------------------------
    model = BabyGOT(cfg, vocab_size=cfg.text.vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] method={cfg.method} device={device} params={n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")
    amp_dtype = _amp_dtype(device)

    def lr_at(step: int) -> float:
        if step < cfg.train.warmup_steps:
            return cfg.train.lr * (step + 1) / cfg.train.warmup_steps
        p = (step - cfg.train.warmup_steps) / max(cfg.train.steps - cfg.train.warmup_steps, 1)
        return cfg.train.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    # ---- loop -------------------------------------------------------------
    model.train()
    step, epoch, running = 0, 0, []
    t0 = time.time()
    while step < cfg.train.steps:
        for images, ids in train_loader:
            if step >= cfg.train.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            images = images.to(device)
            ids = ids.to(device)
            use_amp = cfg.train.amp and device.type == "cuda"
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(images, ids)
            else:
                out = model(images, ids)
            loss = out["loss"] / cfg.train.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % cfg.train.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            running.append(float(loss.detach()) * cfg.train.grad_accum)
            if (step + 1) % cfg.train.log_every == 0:
                extra = {k: float(v.detach()) for k, v in out.items()
                         if k in ("lm_loss", "ot_cost", "gate_reg", "infonce",
                                  "filip_loss")}
                sps = (step + 1) / (time.time() - t0)
                print(f"step {step+1}/{cfg.train.steps} loss={np.mean(running):.4f} "
                      f"sps={sps:.1f} {extra}")
                running = []
            if (step + 1) % cfg.train.eval_every == 0:
                res = quick_eval(model, cfg, tokenizer, device)
                print("[eval]\n" + format_results(res))
                model.train()
            step += 1
        epoch += 1

    # ---- stage 3: instruction tuning ---------------------------------------
    if cfg.train.sft_steps > 0:
        print(f"[sft] instruction tuning for {cfg.train.sft_steps} steps")
        ig = InstructionGenerator(seed=cfg.seed + 77,
                                  image_size=cfg.vision.image_size, device="cpu")
        sft_items = [ig.sample() for _ in range(cfg.train.sft_data)]
        sft_texts = [qa_text(s["question"], s["answer"]) for s in sft_items]
        sft_loader = DataLoader(
            list(zip([s["image"] for s in sft_items], sft_texts)),
            batch_size=cfg.train.batch_size, shuffle=True, drop_last=True,
            collate_fn=lambda b: collate(b, tokenizer, cfg.text.max_len))
        sft_opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.sft_lr,
                                    weight_decay=cfg.train.weight_decay)
        model.train()
        for step, (images, ids) in enumerate(sft_loader):
            if step >= cfg.train.sft_steps:
                break
            images = images.to(device)
            ids = ids.to(device)
            if cfg.train.amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(images, ids, align=False)
            else:
                out = model(images, ids, align=False)
            loss = out["loss"]
            sft_opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(sft_opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(sft_opt)
            scaler.update()
            if (step + 1) % max(cfg.train.sft_steps // 4, 1) == 0:
                print(f"  sft step {step+1}/{cfg.train.sft_steps} "
                      f"loss={float(loss.detach()):.4f}")

    # ---- final evaluation ---------------------------------------------------
    res = quick_eval(model, cfg, tokenizer, device)
    print("[final]\n" + format_results(res))

    # ---- interpretability analysis ------------------------------------------
    try:
        from .analysis import analyze
        an = analyze(model, tokenizer, image_size=cfg.vision.image_size,
                     seed=cfg.seed + 999)
        res["analysis"] = an
        print(f"[analysis] localisation error (patch units) = "
              f"{an['mean_localization_error_patch']:.2f} | "
              f"gate content={an['gate_content']:.3f} function={an['gate_function']:.3f}")
    except Exception as e:  # analysis must never crash a run
        print(f"[analysis] skipped: {e}")

    ckpt = {
        "cfg": cfg, "tokenizer": tokenizer, "model_state": model.state_dict(),
        "results": res,
    }
    torch.save(ckpt, os.path.join(save_dir, f"{cfg.method}.pt"))
    with open(os.path.join(save_dir, f"{cfg.method}.json"), "w") as f:
        json.dump({k: v for k, v in res.items()}, f, indent=2)
    return model, tokenizer, res


def quick_eval(model: BabyGOT, cfg: Config, tokenizer: Tokenizer, device):
    suite = build_suite(n_each=cfg.n_eval_scenes, image_size=cfg.vision.image_size,
                        device="cpu", seed=cfg.seed)
    return evaluate_suite(model, suite, tokenizer)


__all__ = ["train", "seed_everything", "build_tokenizer", "get_device",
           "quick_eval"]
