"""Developmentally-aligned probes (a DevCV-Toolbox-style suite).

Each probe is built *deterministically* from the same scene primitives as the
training data, mirroring how BabyVLM/V2 build benchmarks from SAYCam annotations.
Three scoring modes are supported (see evaluate.py):

  * "text-choice"   : image + question, choose among option strings
                      (Counting, Localization, Left/Right, Who-Has-More,
                       Spatial-details, VTWT-phrase)
  * "image-choice"  : target phrase, choose among candidate images
                      (Picture Vocabulary, VTWT-image)
  * "generation"    : produce a caption for a scene (SAYCam-Caption analogue)

Every item is a plain dict so the probes can be pickled / regenerated identically
with a fixed seed.
"""

from __future__ import annotations

import random
from typing import Dict, List

import torch

from .data import (COLORS, GRID, NUMBERS, POS_LABEL, SHAPE_NOUN, SHAPES, SIZES,
                   SceneGenerator, render_scene)

CHOICE_PROMPT = "answer with one word"


def _single(shape, color, size="big", image_size=96):
    return render_scene([{"shape": shape, "color": color, "size": size,
                          "cell": (1, 1)}], image_size)


def _phrase(shape, color, size=None) -> str:
    noun = SHAPE_NOUN[shape]
    return (f"{size} {color} {noun}" if size else f"a {color} {noun}")


# --------------------------------------------------------------------------- #
# 1. Picture vocabulary  (4-way image choice)
# --------------------------------------------------------------------------- #
def build_picture_vocabulary(n: int, seed: int = 1, image_size: int = 96,
                             device="cpu") -> List[Dict]:
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        shape, color = rng.choice(SHAPES), rng.choice(COLORS)
        target = _single(shape, color, image_size=image_size)
        foils = []
        for _ in range(3):
            s2, c2 = rng.choice(SHAPES), rng.choice(COLORS)
            while (s2, c2) == (shape, color):
                s2, c2 = rng.choice(SHAPES), rng.choice(COLORS)
            foils.append(_single(s2, c2, image_size=image_size))
        images = [target] + foils
        items.append({"name": "pv", "type": "image-choice",
                      "target": _phrase(shape, color),
                      "images": images, "answer": 0})
    return items


# --------------------------------------------------------------------------- #
# 2. Counting  (text choice among number words)
# --------------------------------------------------------------------------- #
def build_counting(n: int, seed: int = 2, image_size: int = 96,
                   device="cpu") -> List[Dict]:
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        k = rng.randint(1, 6)
        shape, color = rng.choice(SHAPES), rng.choice(COLORS)
        cells = rng.sample([(r, c) for r in range(GRID) for c in range(GRID)], k)
        objs = [{"shape": shape, "color": color, "size": "big", "cell": cell}
                for cell in cells]
        img = render_scene(objs, image_size)
        items.append({"name": "counting", "type": "text-choice", "image": img,
                      "prompt": f"how many {SHAPE_NOUN[shape]}s?", "options": NUMBERS,
                      "answer": k - 1})
    return items


# --------------------------------------------------------------------------- #
# 3. Localization (left/right and top/bottom)
# --------------------------------------------------------------------------- #
def build_localization(n: int, seed: int = 3, image_size: int = 96,
                       device="cpu", axis: str = "horizontal") -> List[Dict]:
    rng = random.Random(seed + (0 if axis == "horizontal" else 100))
    items = []
    for _ in range(n):
        shape, color = rng.choice(SHAPES), rng.choice(COLORS)
        # the target is placed on an *extreme* row/column so the question is
        # unambiguous; a distractor sits on the opposite side.
        if axis == "horizontal":
            row = rng.randrange(GRID)
            col = rng.choice([0, 2])
            drow, dcol = row, (2 if col == 0 else 0)
            options = ["left", "right"]
            answer = 0 if col == 0 else 1
        else:
            col = rng.randrange(GRID)
            row = rng.choice([0, 2])
            drow, dcol = (2 if row == 0 else 0), col
            options = ["top", "bottom"]
            answer = 0 if row == 0 else 1
        s2, c2 = rng.choice(SHAPES), rng.choice(COLORS)
        while (s2, c2) == (shape, color):
            s2, c2 = rng.choice(SHAPES), rng.choice(COLORS)
        objs = [
            {"shape": shape, "color": color, "size": "big", "cell": (row, col)},
            {"shape": s2, "color": c2, "size": "big", "cell": (drow, dcol)},
        ]
        img = render_scene(objs, image_size)
        noun = SHAPE_NOUN[shape]
        items.append({"name": "localization", "type": "text-choice", "image": img,
                      "prompt": f"where is the {color} {noun}?", "options": options,
                      "answer": answer})
    return items


# --------------------------------------------------------------------------- #
# 4. Who has more  (left group vs right group)
# --------------------------------------------------------------------------- #
def build_who_has_more(n: int, seed: int = 4, image_size: int = 96,
                       device="cpu") -> List[Dict]:
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        k1, k2 = rng.sample([1, 2, 3], 2)
        shape, color = rng.choice(SHAPES), rng.choice(COLORS)
        left = [{"shape": shape, "color": color, "size": "big", "cell": (r, 0)}
                for r in rng.sample(range(GRID), k1)]
        right = [{"shape": shape, "color": color, "size": "big", "cell": (r, 2)}
                 for r in rng.sample(range(GRID), k2)]
        img = render_scene(left + right, image_size)
        items.append({"name": "whohasmore", "type": "text-choice", "image": img,
                      "prompt": "which side has more?", "options": ["left", "right"],
                      "answer": 0 if k1 > k2 else 1})
    return items


# --------------------------------------------------------------------------- #
# 5. Spatial details (same / different)
# --------------------------------------------------------------------------- #
def build_same_different(n: int, seed: int = 5, image_size: int = 96,
                         device="cpu") -> List[Dict]:
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        shape, color = rng.choice(SHAPES), rng.choice(COLORS)
        size = rng.choice(["big", "small"])
        same = rng.random() < 0.5
        if same:
            o2 = {"shape": shape, "color": color, "size": size}
        else:
            if rng.random() < 0.5:
                c2 = rng.choice([c for c in COLORS if c != color])
                o2 = {"shape": shape, "color": c2, "size": size}
            else:
                s2 = rng.choice([s for s in SHAPES if s != shape])
                o2 = {"shape": s2, "color": color, "size": size}
        objs = [{"shape": shape, "color": color, "size": size, "cell": (1, 0)},
                dict(o2, cell=(1, 2))]
        img = render_scene(objs, image_size)
        items.append({"name": "spatial_details", "type": "text-choice", "image": img,
                      "prompt": "are the two objects the same?",
                      "options": ["yes", "no"], "answer": 0 if same else 1})
    return items


# --------------------------------------------------------------------------- #
# 6. Visual delayed response / memory  (where *was* the object?)
# --------------------------------------------------------------------------- #
def build_memory(n: int, seed: int = 6, image_size: int = 96,
                 device="cpu") -> List[Dict]:
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        shape, color = rng.choice(SHAPES), rng.choice(COLORS)
        col = rng.choice([0, 2])
        # "before" frame on the top row, "after" frame empty
        img = render_scene([{"shape": shape, "color": color, "size": "big",
                             "cell": (0, col)}], image_size)
        noun = SHAPE_NOUN[shape]
        items.append({"name": "memory", "type": "text-choice", "image": img,
                      "prompt": f"where was the {color} {noun} before?",
                      "options": ["left", "right"],
                      "answer": 0 if col == 0 else 1})
    return items


# --------------------------------------------------------------------------- #
# 7. Visual two-word test (VTWT)  — phrase-choice and image-choice variants
# --------------------------------------------------------------------------- #
def build_vtwt_phrase(n: int, seed: int = 7, image_size: int = 96,
                      device="cpu") -> List[Dict]:
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        shape, color = rng.choice(SHAPES), rng.choice(COLORS)
        size = rng.choice(["big", "small"])
        img = _single(shape, color, size=size, image_size=image_size)
        pos = f"{size} {color} {SHAPE_NOUN[shape]}"
        if rng.random() < 0.5:   # colour foil
            c2 = rng.choice([c for c in COLORS if c != color])
            neg = f"{size} {c2} {SHAPE_NOUN[shape]}"
        else:                     # shape foil
            s2 = rng.choice([s for s in SHAPES if s != shape])
            neg = f"{size} {color} {SHAPE_NOUN[s2]}"
        if rng.random() < 0.5:
            options, answer = [pos, neg], 0
        else:
            options, answer = [neg, pos], 1
        items.append({"name": "vtwt_phrase", "type": "text-choice", "image": img,
                      "prompt": "what is this", "options": options,
                      "answer": answer})
    return items


def build_vtwt_image(n: int, seed: int = 8, image_size: int = 96,
                     device="cpu") -> List[Dict]:
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        shape, color = rng.choice(SHAPES), rng.choice(COLORS)
        size = rng.choice(["big", "small"])
        target = _single(shape, color, size=size, image_size=image_size)
        foils = []
        for _ in range(3):
            if rng.random() < 0.5:
                c2 = rng.choice([c for c in COLORS if c != color])
                foils.append(_single(shape, c2, size=size, image_size=image_size))
            else:
                s2 = rng.choice([s for s in SHAPES if s != shape])
                foils.append(_single(s2, color, size=size, image_size=image_size))
        items.append({"name": "vtwt_image", "type": "image-choice",
                      "target": f"{size} {color} {SHAPE_NOUN[shape]}",
                      "images": [target] + foils, "answer": 0})
    return items


# --------------------------------------------------------------------------- #
# 8. Captioning (generation)
# --------------------------------------------------------------------------- #
def build_captioning(n: int, seed: int = 9, image_size: int = 96,
                     device="cpu") -> List[Dict]:
    """Well-posed captioning: a single object, 'what is this' -> its label."""
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        shape, color = rng.choice(SHAPES), rng.choice(COLORS)
        size = rng.choice(SIZES)
        img = _single(shape, color, size=size, image_size=image_size)
        caption = (_phrase(shape, color) if rng.random() < 0.5
                   else f"{size} {color} {SHAPE_NOUN[shape]}")
        items.append({"name": "caption", "type": "generation",
                      "image": img, "target": caption, "prompt": "what is this"})
    return items


# --------------------------------------------------------------------------- #
def build_suite(n_each: int = 50, image_size: int = 96, device="cpu",
                seed: int = 0) -> Dict[str, List[Dict]]:
    """Full probe suite (the "DevCV-lite" benchmark)."""
    return {
        "picture_vocabulary": build_picture_vocabulary(n_each, seed + 1, image_size, device),
        "counting": build_counting(n_each, seed + 2, image_size, device),
        "localization_h": build_localization(n_each, seed + 3, image_size, device, "horizontal"),
        "localization_v": build_localization(n_each, seed + 4, image_size, device, "vertical"),
        "who_has_more": build_who_has_more(n_each, seed + 5, image_size, device),
        "spatial_details": build_same_different(n_each, seed + 6, image_size, device),
        "memory": build_memory(n_each, seed + 7, image_size, device),
        "vtwt_phrase": build_vtwt_phrase(n_each, seed + 8, image_size, device),
        "vtwt_image": build_vtwt_image(n_each, seed + 9, image_size, device),
        "caption": build_captioning(n_each, seed + 10, image_size, device),
    }
