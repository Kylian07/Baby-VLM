"""Loader and zero-shot evaluators for DevCV-Toolbox tasks.

DevCV Toolbox (Wang et al., CVPR 2026) stores every task as a list of LLaVA-style
conversations::

    {"id": ...,
     "conversations": [{"from": "human", "value": "<image>... prompt"},
                       {"from": "gpt",   "value": "A"}],
     "image": ["images/....jpg", ...]}

Two of the ten tasks probe exactly the capacity this work targets, and both can
be scored *zero-shot* from a grounding model without any instruction tuning:

``picture_vocabulary``
    "Touch the image of 'comb'.  (A) the first image (B) ..."  -- word to
    referent.  The published baby model scores 32.4 against a chance floor of
    25.0 and a human ceiling of 91.8.

``localize``
    "Point at the stone.  Is it in the (A) top left ... of the image?" -- word
    to region.  A transport plan gives this for free: take the word's argmax
    slot and map it back to its patch centroid.  No box supervision is used.

The public Ego4D variant of these tasks is on Hugging Face as
``wsashawn/devcv_toolbox_ego4d``; the SAYCam variant is Databrary-gated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# "Touch the image of 'comb'" / 'Which is a stroller?' / "Point at the stone."
_QUOTED = re.compile(r"['\"]([^'\"]+)['\"]")
_POINT_AT = re.compile(r"point at (?:the |a |an )?([a-z][a-z \-]*?)\s*[.?,]", re.I)
_WHICH_IS = re.compile(r"which is (?:a |an |the )?([a-z][a-z \-]*?)\s*[.?]", re.I)

_LETTERS = ["A", "B", "C", "D", "E", "F"]
QUADRANTS = ["top left", "top right", "bottom left", "bottom right"]


@dataclass
class DevCVItem:
    item_id: str
    prompt: str
    answer: str
    images: list[Path]

    @property
    def n_images(self) -> int:
        return len(self.images)

    @property
    def answer_index(self) -> int | None:
        """Index of the gold option, if the answer is a letter."""
        a = self.answer.strip().upper()
        return _LETTERS.index(a) if a in _LETTERS else None


def load_task(root: str | Path, task: str, split: str | None = None) -> list[DevCVItem]:
    """Load one task directory into ``DevCVItem`` records.

    Handles both layouts in the wild: ``<root>/<split>/<task>/data.json`` and
    ``<root>/<task>/data.json``.
    """
    root = Path(root)
    base = root / split / task if split else root / task
    path = base / "data.json"
    if not path.exists():
        raise FileNotFoundError(f"no data.json under {base}")

    items = []
    for rec in json.loads(path.read_text()):
        convs = rec.get("conversations", [])
        human = next((c["value"] for c in convs if c.get("from") == "human"), "")
        gpt = next((c["value"] for c in convs if c.get("from") == "gpt"), "")
        imgs = rec.get("image") or []
        if isinstance(imgs, str):
            imgs = [imgs]
        items.append(
            DevCVItem(
                item_id=str(rec.get("id", len(items))),
                prompt=human,
                answer=str(gpt),
                images=[base / p for p in imgs],
            )
        )
    return items


def target_word(prompt: str) -> str | None:
    """Pull the queried word out of a picture-vocabulary or localization prompt."""
    cleaned = prompt.replace("<image>", " ")
    for pattern in (_QUOTED, _POINT_AT, _WHICH_IS):
        m = pattern.search(cleaned)
        if m:
            return m.group(1).strip().lower()
    return None


def evaluate_picture_vocabulary(items, score_fn) -> dict:
    """Zero-shot picture vocabulary.

    Args:
        items: ``DevCVItem`` list.
        score_fn: ``(word, image_path) -> float``, higher = better match.

    Returns:
        dict with ``accuracy``, ``n``, ``n_skipped`` and the chance floor.
    """
    correct = n = skipped = 0
    chance = []
    for it in items:
        word = target_word(it.prompt)
        gold = it.answer_index
        if word is None or gold is None or gold >= it.n_images:
            skipped += 1
            continue
        scores = [score_fn(word, p) for p in it.images]
        correct += int(max(range(len(scores)), key=scores.__getitem__) == gold)
        chance.append(1.0 / it.n_images)
        n += 1
    return {
        "accuracy": correct / n if n else float("nan"),
        "n": n,
        "n_skipped": skipped,
        "chance": sum(chance) / len(chance) if chance else float("nan"),
    }


def evaluate_localization(items, quadrant_fn) -> dict:
    """Zero-shot localization, scored against the gold quadrant letter.

    Args:
        quadrant_fn: ``(word, image_path) -> str`` returning one of
            ``QUADRANTS``.  A transport-based model implements this by taking
            the word's argmax slot and mapping it to a patch centroid.
    """
    correct = n = skipped = 0
    for it in items:
        word = target_word(it.prompt)
        gold = it.answer_index
        if word is None or gold is None or not it.images or gold >= len(QUADRANTS):
            skipped += 1
            continue
        pred = quadrant_fn(word, it.images[0])
        correct += int(pred == QUADRANTS[gold])
        n += 1
    return {
        "accuracy": correct / n if n else float("nan"),
        "n": n,
        "n_skipped": skipped,
        "chance": 0.25,
    }


def slot_to_quadrant(slot_index: int, grid_h: int, grid_w: int) -> str:
    """Map a patch-grid slot index to one of the four DevCV quadrant labels."""
    row, col = divmod(int(slot_index), grid_w)
    vert = "top" if row < grid_h / 2 else "bottom"
    horiz = "left" if col < grid_w / 2 else "right"
    return f"{vert} {horiz}"
