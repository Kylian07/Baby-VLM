"""Evaluation harness for the developmentally-aligned probe suite.

Scoring rule (identical for every method, so comparisons are fair):
a discriminative probe is answered by *maximum conditional likelihood*:

    answer = argmax_c  log p(c | image, prompt)

For "text-choice" probes this means scoring each option string; for
"image-choice" probes it means scoring each candidate image against the target
phrase.  Generation probes (captioning) are scored with token-F1 vs the ground
truth child-directed utterance.
"""

from __future__ import annotations

import re
from typing import Dict, List

import torch

from .model import BabyGOT
from .tokenizer import Tokenizer


# --------------------------------------------------------------------------- #
# likelihood scorers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _seq_logp(model: BabyGOT, image: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """ids: (B, L).  Returns (B,) sum of log p per sequence."""
    return model.seq_logp(image.to(next(model.parameters()).device), ids)


@torch.no_grad()
def text_scores(model: BabyGOT, image: torch.Tensor, prompt: str,
                options: List[str], tokenizer: Tokenizer) -> torch.Tensor:
    """Image-conditional evidence for each option -> (len(options),).

    A bare next-token likelihood is dominated by *language-model priors*
    (e.g. "left" is a-priori more likely than "right"), which drowns the visual
    signal — the well-known surface-form-competition problem (Holtzman et al.,
    2021).  We therefore score by the *image's contribution* to the answer:

        score(o) = log p(o | image, prompt) - log p(o | blank, prompt)

    i.e. a pointwise-mutual-information / conditional-likelihood ratio, where the
    blank (uniform grey) image supplies the language prior.  The prompt is
    wrapped in the instruction format the model was tuned with ("q: … a:").
    """
    dev = next(model.parameters()).device
    img = image.unsqueeze(0).to(dev) if image.dim() == 3 else image.to(dev)
    blank = torch.full_like(img, 0.5)
    prefix = f"q: {prompt} a:" if prompt else ""
    p_ids = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    p_t = torch.tensor([p_ids], device=dev)
    lp_img = _seq_logp(model, img, p_t).item()
    lp_blank = _seq_logp(model, blank, p_t).item()
    scores = []
    for o in options:
        o_ids = tokenizer.encode(o, add_bos=False, add_eos=True)
        full = torch.tensor([p_ids + o_ids], device=dev)
        s_img = _seq_logp(model, img, full).item() - lp_img
        s_blank = _seq_logp(model, blank, full).item() - lp_blank
        scores.append(s_img - s_blank)
    return torch.tensor(scores)


@torch.no_grad()
def image_scores(model: BabyGOT, phrase: str, images: List[torch.Tensor],
                 tokenizer: Tokenizer) -> torch.Tensor:
    """log p(phrase | image_i, "q: what is this a:") per candidate image.

    Image-choice probes are answered exactly like a naming VQA: the image that
    makes the *answer phrase* most likely under the standard prompt wins.
    """
    dev = next(model.parameters()).device
    prefix = tokenizer.encode("q: what is this a:", add_bos=True, add_eos=False)
    answer = tokenizer.encode(phrase, add_bos=False, add_eos=True)
    p_t = torch.tensor([prefix], device=dev)
    full = torch.tensor([prefix + answer], device=dev)
    scores = []
    for im in images:
        img = im.unsqueeze(0).to(dev) if im.dim() == 3 else im.to(dev)
        logp_prefix = _seq_logp(model, img, p_t).item()
        logp_full = _seq_logp(model, img, full).item()
        scores.append(logp_full - logp_prefix)
    return torch.tensor(scores)


# --------------------------------------------------------------------------- #
# token-F1 (a light METEOR) for captioning
# --------------------------------------------------------------------------- #
_STOP = {"a", "an", "the", "is", "on", "in", "are", "look", "at", "and"}


def _content_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z]+", text.lower()) if t not in _STOP]


def token_f1(ref: str, hyp: str) -> float:
    r, h = _content_tokens(ref), _content_tokens(hyp)
    if not h:
        return 0.0
    common = sum(1 for t in h if t in r)
    p = common / len(h)
    rec = common / len(r) if r else 0.0
    return 2 * p * rec / (p + rec + 1e-9)


# --------------------------------------------------------------------------- #
# per-probe evaluation
# --------------------------------------------------------------------------- #
def _eval_choice(items: List[Dict], model: BabyGOT, tokenizer: Tokenizer,
                 mode: str) -> Dict:
    correct, total = 0, 0
    for it in items:
        if mode == "text-choice":
            sc = text_scores(model, it["image"], it.get("prompt", ""),
                             it["options"], tokenizer)
            pred = int(sc.argmax())
        else:  # image-choice
            sc = image_scores(model, it["target"], it["images"], tokenizer)
            pred = int(sc.argmax())
        correct += (pred == it["answer"])
        total += 1
    return {"correct": correct, "total": total,
            "acc": correct / max(total, 1)}


def _eval_generation(items: List[Dict], model: BabyGOT,
                     tokenizer: Tokenizer, max_new: int = 16) -> Dict:
    dev = next(model.parameters()).device
    f1s = []
    for it in items:
        img = it["image"].unsqueeze(0).to(dev)
        prefix = f"q: {it['prompt']} a:" if it.get("prompt") else "q: what is this a:"
        ids = torch.tensor([tokenizer.encode(prefix, add_bos=True, add_eos=False)],
                           device=dev)
        gen = model.generate(img, ids, max_new=max_new,
                             eos_id=tokenizer.eos_id())
        hyp = tokenizer.decode(gen[0].tolist())
        f1s.append(token_f1(it["target"], hyp))
    f1 = sum(f1s) / max(len(f1s), 1)
    return {"f1": f1, "n": len(f1s)}


def evaluate_suite(model: BabyGOT, suite: Dict[str, List[Dict]],
                   tokenizer: Tokenizer, max_new: int = 16) -> Dict:
    """Evaluate all probes; return per-probe metrics + an overall summary."""
    model.eval()
    results = {}
    choice_accs = []
    for name, items in suite.items():
        if not items:
            continue
        kind = items[0]["type"]
        if kind == "generation":
            r = _eval_generation(items, model, tokenizer, max_new)
            results[name] = r
        else:
            r = _eval_choice(items, model, tokenizer, kind)
            results[name] = r
            choice_accs.append(r["acc"])
    results["overall_choice_acc"] = (sum(choice_accs) / len(choice_accs)
                                     if choice_accs else 0.0)
    return results


def format_results(results: Dict) -> str:
    lines = []
    for k, v in results.items():
        if k == "overall_choice_acc":
            continue
        if "acc" in v:
            lines.append(f"  {k:22s}  acc = {v['acc']:.3f}  ({v['correct']}/{v['total']})")
        else:
            lines.append(f"  {k:22s}  F1  = {v['f1']:.3f}")
    lines.append(f"  {'overall (choice)':22s}  acc = {results['overall_choice_acc']:.3f}")
    return "\n".join(lines)
