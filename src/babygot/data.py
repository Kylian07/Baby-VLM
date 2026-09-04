"""A programmatic, dependency-free "infant-like" scene generator.

SAYCam / BabyView require IRB-gated access, so to make BabyGOT fully reproducible
on any machine (and on a single Kaggle T4) we ship a *synthetic* egocentric-style
data generator: rendered scenes of coloured geometric objects — the same stimulus
family used by the NIH Baby Toolbox and DevCV Toolbox (coloured shapes for
picture-vocabulary, counting, subitizing, spatial-details, ...).

Every scene comes with (image, child-directed utterance, structured facts) so
that benchmarks can be constructed deterministically from the *facts*, exactly as
BabyVLM builds probes from SAYCam annotations.

The public contract for plugging in real data (SAYCam / BabyView) is a single
item: {"image": (3,H,W) float32 in [0,1], "caption": str}.  See ``FromDiskDataset``.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Vocabulary of the "infant world"
# --------------------------------------------------------------------------- #
SHAPES = ["circle", "square", "triangle", "diamond"]
SHAPE_NOUN = {"circle": "ball", "square": "block",
              "triangle": "triangle", "diamond": "diamond"}
COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]
COLOR_RGB = {
    "red": (0.86, 0.16, 0.16), "blue": (0.16, 0.34, 0.80),
    "green": (0.20, 0.66, 0.28), "yellow": (0.92, 0.82, 0.18),
    "purple": (0.55, 0.28, 0.72), "orange": (0.95, 0.55, 0.15),
}
SIZES = ["big", "small"]
NUMBERS = ["one", "two", "three", "four", "five", "six"]
GRID = 3                      # 3 x 3 spatial cells
POS_LABEL = ["left", "middle", "right"]     # columns
ROW_LABEL = ["top", "middle", "bottom"]     # rows


def _mesh(H: int, W: int, device="cpu"):
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device),
                            torch.arange(W, dtype=torch.float32, device=device),
                            indexing="ij")
    return xx, yy


def _falloff(d: torch.Tensor, aa: float = 2.0) -> torch.Tensor:
    """Soft edge: 1 inside, 0 outside, linear over ~2aa pixels."""
    return ((aa - d) / (2 * aa)).clamp(0, 1)


def _circle_mask(xx, yy, cx, cy, r):
    d = torch.hypot(xx - cx, yy - cy) - r
    return _falloff(d)


def _poly_mask(xx, yy, verts: List[Tuple[float, float]]):
    """Signed distance to a CCW convex polygon = max over edges of signed dist."""
    d = None
    for (x0, y0), (x1, y1) in zip(verts, verts[1:] + verts[:1]):
        ex, ey = x1 - x0, y1 - y0
        L = math.hypot(ex, ey)
        s = -(ex * (yy - y0) - ey * (xx - x0)) / L   # <0 inside (CCW)
        d = s if d is None else torch.maximum(d, s)
    return _falloff(-d)


def _shape_mask(shape: str, xx, yy, cx, cy, s):
    if shape == "circle":
        return _circle_mask(xx, yy, cx, cy, s * 0.9)
    if shape == "square":
        return _poly_mask(xx, yy, [(cx - s, cy - s), (cx + s, cy - s),
                                   (cx + s, cy + s), (cx - s, cy + s)])
    if shape == "triangle":
        return _poly_mask(xx, yy, [(cx, cy - s), (cx + s, cy + s * 0.8),
                                   (cx - s, cy + s * 0.8)])
    if shape == "diamond":
        return _poly_mask(xx, yy, [(cx, cy - s), (cx + s, cy),
                                   (cx, cy + s), (cx - s, cy)])
    raise ValueError(shape)


def render_scene(objects: List[Dict], image_size: int = 96,
                 background: Tuple[float, float, float] = (0.86, 0.86, 0.86),
                 device="cpu") -> torch.Tensor:
    """objects: list of dicts {shape, color, size, cell:(row,col)}."""
    H = W = image_size
    xx, yy = _mesh(H, W, device)
    cell = image_size / GRID
    canvas = torch.full((H, W, 3), 0.0, device=device)
    canvas += torch.tensor(background, device=device)
    for o in objects:
        row, col = o["cell"]
        cx = (col + 0.5) * cell
        cy = (row + 0.5) * cell
        s = cell * (0.42 if o["size"] == "big" else 0.26)
        m = _shape_mask(o["shape"], xx, yy, cx, cy, s)
        rgb = torch.tensor(COLOR_RGB[o["color"]], device=device)
        canvas = canvas * (1 - m.unsqueeze(-1)) + rgb * m.unsqueeze(-1)
    return canvas.permute(2, 0, 1).clamp(0, 1)          # (3,H,W)


# --------------------------------------------------------------------------- #
# Scene generator
# --------------------------------------------------------------------------- #
def _plural(noun: str, n: int) -> str:
    return noun if n == 1 else noun + "s"


class SceneGenerator:
    """Samples scenes + child-directed utterances + structured facts."""

    def __init__(self, seed: int = 0, image_size: int = 96,
                 max_objects: int = 3, device="cpu"):
        self.rng = random.Random(seed)
        self.image_size = image_size
        self.max_objects = max_objects
        self.device = device

    # -- sampling ---------------------------------------------------------- #
    def _sample_object(self) -> Dict:
        return {
            "shape": self.rng.choice(SHAPES),
            "color": self.rng.choice(COLORS),
            "size": self.rng.choice(SIZES),
            "cell": (self.rng.randrange(GRID), self.rng.randrange(GRID)),
        }

    def sample(self) -> Dict:
        """Returns {image, caption, facts}."""
        n = self.rng.randint(1, self.max_objects)
        objects = []
        cells = set()
        while len(objects) < n:
            o = self._sample_object()
            if o["cell"] not in cells:
                cells.add(o["cell"])
                objects.append(o)

        kind = self.rng.random()
        target = objects[0]
        noun = SHAPE_NOUN[target["shape"]]

        if kind < 0.5 or n == 1:
            # naming / attribute composition
            if self.rng.random() < 0.5:
                caption = f"a {target['color']} {noun}"
            else:
                caption = f"the {target['size']} {target['color']} {noun}"
        elif kind < 0.75:
            # counting: render k identical objects (re-specialise the scene)
            k = self.rng.randint(1, 6)
            o = self._sample_object()
            objects = [dict(o, cell=cell) for cell in
                       self.rng.sample([(r, c) for r in range(GRID) for c in range(GRID)], k)]
            caption = f"{NUMBERS[k - 1]} {_plural(SHAPE_NOUN[o['shape']], k)}"
            target = objects[0]
        else:
            # localization (with a distractor present)
            col = target["cell"][1]
            caption = f"the {target['color']} {noun} is on the {POS_LABEL[col]}"

        image = render_scene(objects, self.image_size, device=self.device)
        return {
            "image": image,
            "caption": caption,
            "facts": {"objects": objects, "target": target, "caption": caption},
        }

    def sample_batch(self, n: int) -> Dict:
        images, captions, facts = [], [], []
        for _ in range(n):
            s = self.sample()
            images.append(s["image"])
            captions.append(s["caption"])
            facts.append(s["facts"])
        return {"images": torch.stack(images), "captions": captions, "facts": facts}


# --------------------------------------------------------------------------- #
# Instruction tuning (stage 3): (image, question, answer) triples
# --------------------------------------------------------------------------- #
class InstructionGenerator:
    """Generates QA pairs for the instruction-tuning stage (BabyVLM-V2 stage 3).

    Text format:  "q: {question} a: {answer}"
    """

    def __init__(self, seed: int = 0, image_size: int = 96, device="cpu"):
        self.rng = random.Random(seed)
        self.image_size = image_size
        self.device = device

    def sample(self) -> Dict:
        kind = self.rng.random()
        if kind < 0.20:
            return self._naming()
        if kind < 0.40:
            return self._counting()
        if kind < 0.60:
            return self._localization()
        if kind < 0.72:
            return self._who_has_more()
        if kind < 0.84:
            return self._same_different()
        if kind < 0.94:
            return self._memory()
        return self._attribute()

    def _obj(self, cell=None):
        o = {"shape": self.rng.choice(SHAPES), "color": self.rng.choice(COLORS),
             "size": self.rng.choice(SIZES),
             "cell": cell or (self.rng.randrange(GRID), self.rng.randrange(GRID))}
        return o

    def _naming(self):
        o = self._obj((1, 1))
        img = render_scene([o], self.image_size, device=self.device)
        q = "what is this"
        a = f"a {o['color']} {SHAPE_NOUN[o['shape']]}"
        return {"image": img, "question": q, "answer": a}

    def _attribute(self):
        o = self._obj((1, 1))
        img = render_scene([o], self.image_size, device=self.device)
        q = "what is this"
        a = f"{o['size']} {o['color']} {SHAPE_NOUN[o['shape']]}"
        return {"image": img, "question": q, "answer": a}

    def _counting(self):
        k = self.rng.randint(1, 6)
        o = self._obj()
        cells = self.rng.sample([(r, c) for r in range(GRID) for c in range(GRID)], k)
        objs = [dict(o, cell=c) for c in cells]
        img = render_scene(objs, self.image_size, device=self.device)
        q = f"how many {SHAPE_NOUN[o['shape']]}s"
        a = NUMBERS[k - 1]
        return {"image": img, "question": q, "answer": a}

    def _localization(self):
        o = self._obj()
        col = o["cell"][1]
        if col == 1:
            o["cell"] = (o["cell"][0], self.rng.choice([0, 2]))
            col = o["cell"][1]
        d = self._obj((o["cell"][0], 0 if col == 2 else 2))
        img = render_scene([o, d], self.image_size, device=self.device)
        q = f"where is the {o['color']} {SHAPE_NOUN[o['shape']]}"
        a = POS_LABEL[col]
        return {"image": img, "question": q, "answer": a}

    def _who_has_more(self):
        k1, k2 = self.rng.sample([1, 2, 3], 2)
        o = self._obj()
        left = [dict(o, cell=(r, 0)) for r in self.rng.sample(range(GRID), k1)]
        right = [dict(o, cell=(r, 2)) for r in self.rng.sample(range(GRID), k2)]
        img = render_scene(left + right, self.image_size, device=self.device)
        return {"image": img, "question": "which side has more",
                "answer": "left" if k1 > k2 else "right"}

    def _same_different(self):
        o = self._obj()
        same = self.rng.random() < 0.5
        if same:
            o2 = dict(o, cell=(1, 2))
        elif self.rng.random() < 0.5:
            o2 = dict(o, color=self.rng.choice([c for c in COLORS if c != o["color"]]),
                      cell=(1, 2))
        else:
            o2 = dict(o, shape=self.rng.choice([s for s in SHAPES if s != o["shape"]]),
                      cell=(1, 2))
        o["cell"] = (1, 0)
        img = render_scene([o, o2], self.image_size, device=self.device)
        return {"image": img, "question": "are the two objects the same",
                "answer": "yes" if same else "no"}

    def _memory(self):
        o = self._obj()
        col = self.rng.choice([0, 2])
        o["cell"] = (0, col)
        img = render_scene([o], self.image_size, device=self.device)
        q = f"where was the {o['color']} {SHAPE_NOUN[o['shape']]} before"
        return {"image": img, "question": q, "answer": POS_LABEL[col]}


def qa_text(question: str, answer: str) -> str:
    return f"q: {question} a: {answer}"


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class SceneDataset(Dataset):
    def __init__(self, n: int, generator: SceneGenerator):
        self.items = [generator.sample() for _ in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]["image"], self.items[i]["caption"]


class FromDiskDataset(Dataset):
    """Adapter for real (SAYCam/BabyView) data: a folder of (img, txt) pairs.

    Use ``make_disk_dataset`` to dump a list of dicts.  Contract per item:
    {"image": (3,H,W) float32 in [0,1], "caption": str}.
    """

    def __init__(self, items: List[Dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]["image"], self.items[i]["caption"]


def make_disk_dataset(images: List[torch.Tensor], captions: List[str]) -> FromDiskDataset:
    return FromDiskDataset([{"image": im, "caption": c}
                            for im, c in zip(images, captions)])


def collate(batch, tokenizer, max_len: int):
    """(list of (image, caption)) -> (images, ids).  Pads to max_len."""
    images, captions = zip(*batch)
    images = torch.stack([im for im in images])
    ids = [tokenizer.encode(c) for c in captions]
    L = min(max_len, max(len(i) for i in ids))
    buf = torch.full((len(ids), L), tokenizer.pad_id(), dtype=torch.long)
    for i, row in enumerate(ids):
        row = row[:L]
        buf[i, :len(row)] = torch.tensor(row, dtype=torch.long)
    return images, buf
