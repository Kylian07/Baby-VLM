"""Image-utterance corpora and the tokenizer used for referential alignment.

The alignment loss operates on *content words*, not on every token: function
words ("the", "is", "there") have no referent, and including them just enlarges
the transport problem.  ``WordTokenizer`` therefore exposes two views of an
utterance -- the full token sequence for the captioning objective, and the
content-word subset for the alignment objective.

Backends, in order of how easy they are to obtain:

``synthetic``
    Procedurally generated scenes and utterances.  Needs no downloads, so the
    whole training pipeline is runnable and testable immediately, and the
    referential-ambiguity parameters are known exactly.
``jsonl`` / ``json``
    Any ``{"image": path, "text": utterance}`` records.  This is the adapter for
    COCO, Localized Narratives, or anything else already on disk.
``saycam``
    Frames plus transcribed caregiver speech.  Databrary-gated; included so the
    switch is a config change, not a rewrite.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# Deliberately short: a closed-class list, not a topic-specific stoplist.
STOPWORDS = set(
    """a an the this that these those there here it its is are was were be been am
    do does did doing done have has had having will would can could shall should may
    might must of in on at to for with from by about into over under again then once
    and or but if because as until while so than too very just now not no nor only own
    same s t don now i you he she we they me him her us them my your his our their what
    which who whom when where why how all any both each few more most other some such
    up down out off further oh ok okay yeah yes uh um hey look see got get go going""".split()
)

_TOKEN = re.compile(r"[a-z']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class WordTokenizer:
    """Word-level vocabulary.  Child-directed speech has a small lexicon."""

    itos: list[str]
    stoi: dict[str, int]
    max_len: int = 32
    max_content: int = 8

    @classmethod
    def build(cls, texts, max_vocab: int = 8192, min_count: int = 2, **kw) -> "WordTokenizer":
        counts = Counter(w for t in texts for w in tokenize(t))
        keep = [w for w, c in counts.most_common(max_vocab - len(SPECIALS)) if c >= min_count]
        itos = SPECIALS + keep
        return cls(itos=itos, stoi={w: i for i, w in enumerate(itos)}, **kw)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        ids = [self.stoi.get(w, UNK) for w in tokenize(text)][: self.max_len - 2]
        return [BOS] + ids + [EOS]

    def content_ids(self, text: str) -> list[int]:
        """Deduplicated content words, in order of first appearance."""
        out, seen = [], set()
        for w in tokenize(text):
            if w in STOPWORDS or len(w) < 2:
                continue
            i = self.stoi.get(w, UNK)
            if i == UNK or i in seen:
                continue
            seen.add(i)
            out.append(i)
            if len(out) >= self.max_content:
                break
        return out

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"itos": self.itos, "max_len": self.max_len,
                                          "max_content": self.max_content}))

    @classmethod
    def load(cls, path) -> "WordTokenizer":
        d = json.loads(Path(path).read_text())
        return cls(itos=d["itos"], stoi={w: i for i, w in enumerate(d["itos"])},
                   max_len=d["max_len"], max_content=d["max_content"])


class PairDataset(Dataset):
    """(image, utterance) pairs from a list of records."""

    def __init__(self, records, tokenizer: WordTokenizer, image_size: int = 128, root=None):
        self.records = records
        self.tok = tokenizer
        self.image_size = image_size
        self.root = Path(root) if root else None

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path) -> torch.Tensor:
        from PIL import Image

        p = self.root / path if self.root else Path(path)
        im = Image.open(p).convert("RGB").resize((self.image_size, self.image_size))
        a = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (a - 0.5) / 0.5

    def __getitem__(self, i):
        rec = self.records[i]
        text = rec["text"]
        return {
            "image": self._load_image(rec["image"]),
            "tokens": torch.tensor(self.tok.encode(text), dtype=torch.long),
            "content": torch.tensor(self.tok.content_ids(text), dtype=torch.long),
        }


def collate(batch, pad: int = PAD):
    """Pad tokens and content words, returning masks for both."""
    n = len(batch)
    tl = max(len(b["tokens"]) for b in batch)
    cl = max(max(len(b["content"]) for b in batch), 1)
    tokens = torch.full((n, tl), pad, dtype=torch.long)
    tmask = torch.zeros(n, tl, dtype=torch.bool)
    content = torch.full((n, cl), pad, dtype=torch.long)
    cmask = torch.zeros(n, cl, dtype=torch.bool)
    for i, b in enumerate(batch):
        t, c = b["tokens"], b["content"]
        tokens[i, : len(t)] = t
        tmask[i, : len(t)] = True
        content[i, : len(c)] = c
        cmask[i, : len(c)] = True
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "tokens": tokens,
        "token_mask": tmask,
        "content": content,
        "content_mask": cmask,
    }


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def load_records(spec: str, root=None, limit: int | None = None) -> list[dict]:
    """Load ``{"image", "text"}`` records from a json/jsonl file."""
    path = Path(spec)
    if path.suffix == ".jsonl":
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    else:
        recs = json.loads(path.read_text())
    out = []
    for r in recs:
        img = r.get("image") or r.get("image_path") or r.get("file_name")
        txt = r.get("text") or r.get("caption") or r.get("utterance")
        if isinstance(img, list):
            img = img[0]
        if img and txt:
            out.append({"image": img, "text": txt})
    return out[:limit] if limit else out


class SyntheticScenes(Dataset):
    """Procedural scenes with known ground truth, matching ``gavagai.sim``.

    Each scene renders a few coloured shapes on a canvas and produces an
    utterance naming some of them, with configurable rates of non-referential
    speech and of naming an object that is not present.  Because the
    word-referent mapping is known, picture-vocabulary accuracy is measurable
    without any external benchmark -- which makes this the right smoke test for
    the whole pipeline before spending GPU hours on real data.
    """

    COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "brown"]
    SHAPES = ["ball", "block", "cup", "star", "ring", "bar"]
    FILLER = "look at that see it here we go is nice now oh yes okay come on".split()

    def __init__(self, n: int = 4000, image_size: int = 128, null_rate: float = 0.6,
                 absent_rate: float = 0.4, n_objects: int = 3, seed: int = 0):
        self.n = n
        self.image_size = image_size
        self.null_rate = null_rate
        self.absent_rate = absent_rate
        self.n_objects = n_objects
        self.seed = seed
        self.names = [f"{c} {s}" for c in self.COLORS for s in self.SHAPES]
        self.words = [f"{c}{s}" for c in self.COLORS for s in self.SHAPES]

    def __len__(self) -> int:
        return self.n

    def _render(self, rng, ids):
        img = np.full((self.image_size, self.image_size, 3), 0.15, dtype=np.float32)
        size = self.image_size // 4
        rgb = {
            "red": (0.9, 0.1, 0.1), "green": (0.1, 0.8, 0.2), "blue": (0.15, 0.3, 0.9),
            "yellow": (0.95, 0.9, 0.1), "purple": (0.6, 0.15, 0.8),
            "orange": (0.95, 0.55, 0.1), "pink": (0.95, 0.5, 0.7), "brown": (0.45, 0.28, 0.1),
        }
        cells = rng.choice(9, size=len(ids), replace=False)
        for oid, cell in zip(ids, cells):
            color = self.COLORS[oid // len(self.SHAPES)]
            shape = oid % len(self.SHAPES)
            r, c = divmod(int(cell), 3)
            y0 = r * (self.image_size // 3) + 4
            x0 = c * (self.image_size // 3) + 4
            patch = np.zeros((size, size, 3), dtype=np.float32)
            yy, xx = np.mgrid[0:size, 0:size]
            cy = cx = size / 2
            if shape % 3 == 0:
                m = (yy - cy) ** 2 + (xx - cx) ** 2 <= (size / 2.2) ** 2
            elif shape % 3 == 1:
                m = np.ones((size, size), dtype=bool)
            else:
                m = np.abs(yy - cy) + np.abs(xx - cx) <= size / 2.2
            patch[m] = rgb[color]
            patch += 0.05 * rng.standard_normal(patch.shape).astype(np.float32)
            img[y0:y0 + size, x0:x0 + size] = np.clip(patch, 0, 1)
        return img

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        present = rng.choice(len(self.words), size=self.n_objects, replace=False)
        img = self._render(rng, present)

        spoken = []
        for oid in present:
            if rng.random() < self.null_rate:
                spoken.append(str(rng.choice(self.FILLER)))
            else:
                spoken.append(self.words[oid])
        if rng.random() < self.absent_rate:
            absent = [j for j in range(len(self.words)) if j not in set(present.tolist())]
            spoken.append(self.words[int(rng.choice(absent))])
        rng.shuffle(spoken)

        x = torch.from_numpy(img).permute(2, 0, 1)
        return {"image": (x - 0.5) / 0.5, "text": " ".join(spoken),
                "present": torch.as_tensor(present.copy())}

    def texts(self):
        return [self[i]["text"] for i in range(min(self.n, 2000))]

    def probe_images(self, seed: int = 12345) -> torch.Tensor:
        """One canonical single-object image per word, for picture-vocabulary eval.

        Held out from training by construction: these scenes contain exactly one
        object and are rendered from a disjoint seed stream.
        """
        out = []
        for oid in range(len(self.words)):
            rng = np.random.default_rng(seed + oid)
            img = self._render(rng, [oid])
            x = torch.from_numpy(img).permute(2, 0, 1)
            out.append((x - 0.5) / 0.5)
        return torch.stack(out)
