"""Loader and zero-shot evaluators for DevCV-Toolbox tasks.

DevCV Toolbox (Wang et al., CVPR 2026) ships in more than one directory layout,
so discovery here is deliberately layout-agnostic.  The public Hugging Face
release (``wsashawn/devcv_toolbox_ego4d``) is **flat**::

    data/Ego4D/
      pv_test.json  leftright_test.json  localize_test.json  ...
      images/

while the samples published on the workshop website nest one directory per
task::

    <root>/[<split>/]<task>/data.json

Both are handled by :func:`find_tasks`.  Image paths inside a record are
resolved against several candidate roots, because whether they are relative to
the JSON file or to the dataset root differs between the two layouts.

Every task is a list of LLaVA-style conversations::

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
from dataclasses import dataclass, field
from pathlib import Path

# "Touch the image of 'comb'" / 'Which is a stroller?' / "Point at the stone."
_QUOTED = re.compile(r"['\"]([^'\"]+)['\"]")
_POINT_AT = re.compile(r"point at (?:the |a |an )?([a-z][a-z \-]*?)\s*[.?,]", re.I)
_WHICH_IS = re.compile(r"which is (?:a |an |the )?([a-z][a-z \-]*?)\s*[.?]", re.I)

_LETTERS = ["A", "B", "C", "D", "E", "F"]
QUADRANTS = ["top left", "top right", "bottom left", "bottom right"]

# The flat release abbreviates some task names relative to the website samples.
# Callers may use either spelling.
TASK_ALIASES = {
    "pv": "picture_vocabulary",
    "picture_vocab": "picture_vocabulary",
    "picture_vocab_selected": "picture_vocabulary",
    "vdr_binary": "vdr-binary",
    "vdr_open": "vdr-open",
    "localise": "localize",
    "spatial": "spatialdetails",
}


def canonical_task(name: str) -> str:
    """Map a task name onto its canonical spelling."""
    return TASK_ALIASES.get(name, name)


@dataclass
class TaskSource:
    """One task's JSON file plus where to look for its images."""

    name: str
    path: Path
    image_roots: list[Path]
    split: str | None = None

    @property
    def canonical(self) -> str:
        return canonical_task(self.name)


def find_tasks(root: str | Path) -> dict[str, list[TaskSource]]:
    """Discover every DevCV task under ``root``, whichever layout it uses.

    Returns a mapping from canonical task name to the sources found for it.
    """
    root = Path(root)
    out: dict[str, list[TaskSource]] = {}

    def add(src: TaskSource) -> None:
        out.setdefault(src.canonical, []).append(src)

    def _hidden(path: Path) -> bool:
        """Skip hidden trees. The Hugging Face downloader leaves a .cache/
        directory beside the data, and anything inside it is bookkeeping."""
        return any(part.startswith(".") for part in path.relative_to(root).parts[:-1])

    # Layout A: <root>/[<split>/]<task>/data.json
    for path in sorted(root.rglob("data.json")):
        if _hidden(path):
            continue
        task = path.parent.name
        parent = path.parent.parent
        split = parent.name if parent.name in {"train", "val", "test"} else None
        add(TaskSource(task, path, _image_roots(path, root), split))

    # Layout B: <root>/<task>[_<split>].json  (the public Hugging Face release)
    for path in sorted(root.rglob("*.json")):
        if path.name == "data.json" or _hidden(path) or path.name.startswith("."):
            continue
        stem = path.stem
        split = None
        for suffix in ("_test", "_val", "_train"):
            if stem.endswith(suffix):
                stem, split = stem[: -len(suffix)], suffix[1:]
                break
        add(TaskSource(stem, path, _image_roots(path, root), split))

    return out


def _image_roots(json_path: Path, root: Path) -> list[Path]:
    """Candidate bases for resolving a record's relative image paths.

    Ordered most- to least-specific; the first one where the file actually
    exists wins.  This is what makes the loader work for both layouts without
    the caller having to know which it has.
    """
    cands = [json_path.parent, json_path.parent.parent, root]
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


_INDEX_CACHE: dict[Path, dict[str, list[Path]]] = {}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _basename_index(root: Path) -> dict[str, list[Path]]:
    """Map every image basename under ``root`` to the paths carrying it.

    Built once per root and cached.  Used only as a fallback, and only when the
    basename is unambiguous -- see :func:`_resolve`.
    """
    if root not in _INDEX_CACHE:
        idx: dict[str, list[Path]] = {}
        for p in root.rglob("*"):
            if p.suffix.lower() in _IMAGE_SUFFIXES and p.is_file():
                idx.setdefault(p.name, []).append(p)
        _INDEX_CACHE[root] = idx
    return _INDEX_CACHE[root]


def _resolve(rel: str, roots: list[Path]) -> Path:
    """Locate a record's image.

    Tries the relative path against each candidate root, then falls back to
    matching on basename alone.  The fallback deliberately refuses to resolve an
    **ambiguous** basename: silently picking one of several same-named files
    could make a matching task look solvable when it is not, which would
    fabricate a result rather than merely miss one.
    """
    for r in roots:
        p = r / rel
        if p.exists():
            return p

    name = Path(rel).name
    for r in roots:
        hits = _basename_index(r).get(name, [])
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            # Ambiguous: prefer a hit whose tail matches more of the given path.
            parts = tuple(Path(rel).parts)
            exact = [h for h in hits if tuple(h.parts[-len(parts):]) == parts]
            if len(exact) == 1:
                return exact[0]
            break  # genuinely ambiguous -- do not guess

    return roots[0] / rel  # non-existent; caller reports it as unresolved


@dataclass
class DevCVItem:
    item_id: str
    prompt: str
    answer: str
    images: list[Path]
    raw_images: list[str] = field(default_factory=list)
    """The image strings exactly as they appear in the JSON, before resolution."""

    @property
    def n_images(self) -> int:
        return len(self.images)

    @property
    def answer_index(self) -> int | None:
        """Index of the gold option, if the answer is a letter."""
        a = self.answer.strip().upper()
        return _LETTERS.index(a) if a in _LETTERS else None


def load_source(src: TaskSource) -> list[DevCVItem]:
    """Read one discovered task file into ``DevCVItem`` records."""
    items = []
    payload = json.loads(src.path.read_text())
    if isinstance(payload, dict):  # some dumps wrap the list
        payload = payload.get("data") or payload.get("items") or []
    for rec in payload:
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
                images=[_resolve(p, src.image_roots) for p in imgs],
                raw_images=list(imgs),
            )
        )
    return items


def load_task(root: str | Path, task: str, split: str | None = None) -> list[DevCVItem]:
    """Load one task by name, from whichever layout ``root`` happens to use."""
    tasks = find_tasks(root)
    want = canonical_task(task)
    sources = tasks.get(want, [])
    if split is not None:
        sources = [s for s in sources if s.split == split] or sources
    if not sources:
        raise FileNotFoundError(
            f"task {task!r} not found under {root}. Discovered: {sorted(tasks)}"
        )
    items: list[DevCVItem] = []
    for s in sources:
        items.extend(load_source(s))
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
