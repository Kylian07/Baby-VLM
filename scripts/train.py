#!/usr/bin/env python3
"""Train a from-scratch baby VLM, with or without referential alignment.

The baseline is LLaVA-style autoregressive captioning -- the objective every
stage of BabyVLM-V2 uses.  ``--aux-weight > 0`` adds the GAVAGAI alignment term
on top of the *identical* captioning loss and the *identical* parameters, so the
comparison isolates one added term.

Sized for a Kaggle T4: ~20M parameters, fp16 autocast, and checkpoint/resume so
a run survives the 12-hour session limit.

Examples
--------
Smoke test with no downloads at all (verifies the whole pipeline end to end)::

    python scripts/train.py --data synthetic --steps 200 --eval-every 100

The headline comparison::

    python scripts/train.py --data synthetic --steps 4000 --aux-weight 0.0  --out runs/ar
    python scripts/train.py --data synthetic --steps 4000 --aux-weight 1.0 \
        --rho 1.0 --use-null --out runs/gavagai

Real corpus (any ``{"image","text"}`` json/jsonl; COCO, Localized Narratives, ...)::

    python scripts/train.py --data pairs.jsonl --image-root /path/to/images --steps 8000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gavagai.data.corpora import (
    PAD,
    PairDataset,
    SyntheticScenes,
    WordTokenizer,
    collate,
    load_records,
)
from gavagai.losses import gavagai_loss
from gavagai.models import BabyVLM, ModelConfig


def build_data(args):
    if args.data == "synthetic":
        scenes = SyntheticScenes(
            n=args.synthetic_n, image_size=args.image_size,
            null_rate=args.null_rate, absent_rate=args.absent_rate,
        )
        tok = WordTokenizer.build(scenes.texts(), min_count=1, max_len=args.max_len)
        records = [{"image": None, "text": None}] * 0  # unused

        class _Wrap(torch.utils.data.Dataset):
            def __len__(self):
                return len(scenes)

            def __getitem__(self, i):
                s = scenes[i]
                return {
                    "image": s["image"],
                    "tokens": torch.tensor(tok.encode(s["text"]), dtype=torch.long),
                    "content": torch.tensor(tok.content_ids(s["text"]), dtype=torch.long),
                }

        return _Wrap(), tok, scenes

    records = load_records(args.data, limit=args.limit)
    if not records:
        raise SystemExit(f"no usable records in {args.data}")
    tok = WordTokenizer.build((r["text"] for r in records), max_len=args.max_len)
    ds = PairDataset(records, tok, image_size=args.image_size, root=args.image_root)
    return ds, tok, None


@torch.no_grad()
def synthetic_pv_accuracy(model, scenes, tok, device, batch: int = 32) -> float:
    """Picture-vocabulary probe: match each word to its single-object image."""
    model.eval()
    probes = scenes.probe_images().to(device)
    slots = torch.cat([model.encoder.encode_slots(probes[i:i + batch])
                       for i in range(0, len(probes), batch)])
    img_vec = F.normalize(slots.max(1).values, dim=-1)  # best-matching slot per image
    ids = torch.tensor([tok.stoi.get(w, 3) for w in scenes.words], device=device)
    known = ids != 3
    words = model.word_embeddings_for_alignment(ids.unsqueeze(0))[0]
    pred = (words @ img_vec.t()).argmax(1)
    gold = torch.arange(len(scenes.words), device=device)
    model.train()
    return float((pred[known] == gold[known]).float().mean()) if known.any() else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--synthetic-n", type=int, default=8000)
    ap.add_argument("--null-rate", type=float, default=0.6)
    ap.add_argument("--absent-rate", type=float, default=0.4)

    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)

    ap.add_argument("--aux-weight", type=float, default=0.0)
    ap.add_argument("--rho", type=float, default=1.0, help="-1 means balanced (rho=inf)")
    ap.add_argument("--use-null", action="store_true")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--eps-final", type=float, default=None, help="anneal eps to this value")
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--null-prior", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.07)

    ap.add_argument("--out", type=Path, default=Path("runs/default"))
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    ds, tok, scenes = build_data(args)
    cfg = ModelConfig(image_size=args.image_size, vocab_size=len(tok))
    model = BabyVLM(cfg).to(device)
    tok.save(args.out / "tokenizer.json")

    rho = None if args.rho < 0 else args.rho
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                        num_workers=args.workers, collate_fn=collate, persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    start = 0
    if args.resume and args.resume.exists():
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start = state["step"]
        print(f"resumed from {args.resume} at step {start}")

    print(f"device={device}  params={model.n_parameters()/1e6:.1f}M  vocab={len(tok)}  "
          f"aux_weight={args.aux_weight} rho={rho} use_null={args.use_null}")

    history, step, t0 = [], start, time.time()
    it = iter(loader)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        for gparam in opt.param_groups:
            gparam["lr"] = lr

        images = batch["images"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        content = batch["content"].to(device, non_blocking=True)
        cmask = batch["content_mask"].to(device, non_blocking=True)

        eps = args.eps
        if args.eps_final is not None:
            frac = step / max(1, args.steps - 1)
            eps = args.eps * (args.eps_final / args.eps) ** frac

        with torch.amp.autocast("cuda", enabled=device == "cuda", dtype=torch.float16):
            slots = model.encoder.encode_slots(images)
            logits = model.decoder(slots, tokens[:, :-1])
            ar_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                tokens[:, 1:].reshape(-1),
                ignore_index=PAD,
            )
            loss, aux_val = ar_loss, torch.zeros((), device=device)
            if args.aux_weight > 0 and cmask.any():
                words = model.word_embeddings_for_alignment(content)
                smask = torch.ones(slots.shape[:2], dtype=torch.bool, device=device)
                aux, stats = gavagai_loss(
                    words.float(), slots.float(), cmask, smask,
                    kappa=args.kappa, eps=eps, rho=rho, tau=args.tau,
                    null_prior=args.null_prior, use_null=args.use_null,
                )
                aux_val = aux.detach()
                loss = ar_loss + args.aux_weight * aux

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1

        if step % 50 == 0:
            print(f"step {step:>6}  ar {ar_loss.item():.3f}  aux {aux_val.item():.3f}  "
                  f"lr {lr:.2e}  {step/(time.time()-t0+1e-9):.1f} it/s", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            rec = {"step": step, "ar_loss": float(ar_loss.item())}
            if scenes is not None:
                rec["pv_accuracy"] = synthetic_pv_accuracy(model, scenes, tok, device)
                print(f"  [eval] step {step}  synthetic PV accuracy = {rec['pv_accuracy']:.3f} "
                      f"(chance = {1/len(scenes.words):.3f})", flush=True)
            history.append(rec)
            (args.out / "history.json").write_text(json.dumps(history, indent=2))

        if step % args.ckpt_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "args": vars(args) | {"out": str(args.out)}},
                       args.out / "ckpt.pt")

    print(f"done in {time.time()-t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
