"""Tests for the DevCV-Toolbox loader and the text-blind audit."""
from pathlib import Path

import pytest

from gavagai.data.devcv import (
    QUADRANTS,
    evaluate_localization,
    evaluate_picture_vocabulary,
    load_task,
    slot_to_quadrant,
    target_word,
)

FIX = Path(__file__).parent / "fixtures"
# fixtures/ holds BOTH layouts, so tests must name the one they mean.
NESTED = FIX / "test"      # <root>/<task>/data.json (website samples)
FLAT_ROOT = FIX / "flat"   # <task>_test.json + images/ (HF release)


def test_load_and_parse():
    items = load_task(NESTED, "picture_vocabulary")
    assert len(items) == 1
    it = items[0]
    assert it.n_images == 4
    assert it.answer_index == 1
    assert target_word(it.prompt) == "comb"


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("<image> Touch the image of 'stroller' (A) ...", "stroller"),
        ("<image> Point at the stone. Is it in the (A) top left", "stone"),
        ("<image> Which is a chair? Answer A or B.", "chair"),
        ("<image> How many steps do you see?", None),
    ],
)
def test_target_word_patterns(prompt, expected):
    assert target_word(prompt) == expected


def test_picture_vocabulary_scoring_is_exact():
    items = load_task(NESTED, "picture_vocabulary")
    # An oracle scorer that always prefers the gold image must score 1.0.
    gold_path = items[0].images[1]
    res = evaluate_picture_vocabulary(items, lambda w, p: 1.0 if p == gold_path else 0.0)
    assert res["accuracy"] == 1.0 and res["n"] == 1 and res["chance"] == 0.25
    # A scorer that always prefers a distractor must score 0.0.
    bad = items[0].images[0]
    res = evaluate_picture_vocabulary(items, lambda w, p: 1.0 if p == bad else 0.0)
    assert res["accuracy"] == 0.0


def test_localization_scoring():
    items = load_task(NESTED, "picture_vocabulary")
    gold = items[0].answer_index  # answer "B" -> index 1 -> "top right"
    assert QUADRANTS[gold] == "top right"
    res = evaluate_localization(items, lambda w, p: QUADRANTS[gold])
    assert res["n"] == 1 and res["accuracy"] == 1.0
    wrong = QUADRANTS[(gold + 1) % len(QUADRANTS)]
    res = evaluate_localization(items, lambda w, p: wrong)
    assert res["accuracy"] == 0.0


def test_slot_to_quadrant_covers_the_grid():
    assert slot_to_quadrant(0, 4, 4) == "top left"
    assert slot_to_quadrant(3, 4, 4) == "top right"
    assert slot_to_quadrant(12, 4, 4) == "bottom left"
    assert slot_to_quadrant(15, 4, 4) == "bottom right"


def test_text_blind_matcher_finds_the_duplicate():
    """The audit's core claim in miniature: an exact copy is found without text."""
    from scripts.text_blind_audit import audit_task, image_feature

    items = load_task(NESTED, "leftright")
    feats = [image_feature(p) for p in items[0].images]
    sims = [float(feats[0] @ f) for f in feats[1:]]
    assert int(max(range(3), key=sims.__getitem__)) == items[0].answer_index

    from gavagai.data.devcv import find_tasks

    res = audit_task(find_tasks(NESTED)["leftright"])
    assert res["match_acc"] == 1.0


# ---------------------------------------------------------------------------
# Layout handling.
#
# The public Hugging Face release (wsashawn/devcv_toolbox_ego4d) is FLAT --
# <task>_test.json beside a shared images/ directory -- while the workshop
# website samples nest one directory per task. An earlier version of this loader
# only globbed for data.json, so against the real release it discovered nothing
# and the audit printed an empty table without erroring. These tests pin both.
# ---------------------------------------------------------------------------

FLAT = FLAT_ROOT


def test_discovers_the_flat_huggingface_layout():
    from gavagai.data.devcv import find_tasks

    tasks = find_tasks(FLAT)
    assert "leftright" in tasks
    assert "picture_vocabulary" in tasks, "pv_test.json must map to picture_vocabulary"
    assert tasks["leftright"][0].split == "test"


def test_flat_layout_resolves_images_against_the_dataset_root():
    """Records say 'images/pv/0.jpeg', relative to the root, not to the JSON."""
    items = load_task(FLAT, "pv")
    assert len(items) == 1
    assert all(p.exists() for p in items[0].images), items[0].images


def test_either_task_spelling_loads_the_same_data():
    assert len(load_task(FLAT, "pv")) == len(load_task(FLAT, "picture_vocabulary"))


def test_nested_layout_still_works():
    items = load_task(NESTED, "leftright")
    assert len(items) == 1 and all(p.exists() for p in items[0].images)


def test_missing_task_names_what_was_actually_found():
    """A bad task name should report the alternatives, not just fail."""
    import pytest

    with pytest.raises(FileNotFoundError) as e:
        load_task(FLAT, "no_such_task")
    assert "picture_vocabulary" in str(e.value)


def test_hidden_trees_are_excluded_from_discovery(tmp_path):
    """The Hugging Face downloader leaves a .cache/ tree beside the data.

    Anything in it is bookkeeping and must never be mistaken for a task, or the
    audit would report a task that does not exist.
    """
    import json as _json

    from gavagai.data.devcv import find_tasks

    (tmp_path / "images").mkdir()
    (tmp_path / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (tmp_path / ".cache" / "huggingface" / "download" / "bogus_test.json").write_text("[]")
    (tmp_path / "leftright_test.json").write_text(_json.dumps([{
        "id": "x", "image": [], "conversations": [
            {"from": "human", "value": "(A) (B)"}, {"from": "gpt", "value": "A"}]}]))

    tasks = find_tasks(tmp_path)
    assert set(tasks) == {"leftright"}, tasks


def test_basename_fallback_resolves_unique_files(tmp_path):
    """Records sometimes store a bare filename while images sit in subdirs."""
    import json as _json

    from gavagai.data.devcv import load_task

    (tmp_path / "images" / "leftright").mkdir(parents=True)
    for n in ("q.jpg", "a.jpg"):
        (tmp_path / "images" / "leftright" / n).write_bytes(b"x")
    (tmp_path / "t_test.json").write_text(_json.dumps([{
        "id": "0", "image": ["q.jpg", "a.jpg"],          # bare names, no directory
        "conversations": [{"from": "human", "value": "(A) (B)"},
                          {"from": "gpt", "value": "A"}]}]))
    items = load_task(tmp_path, "t")
    assert all(p.exists() for p in items[0].images), items[0].images


def test_ambiguous_basenames_are_refused_not_guessed(tmp_path):
    """Two files with the same name in different directories must NOT resolve.

    Guessing one would let a matching task appear solvable when it is not --
    a fabricated result, which is worse than a missing one.
    """
    import json as _json

    from gavagai.data.devcv import load_task

    for sub in ("optA", "optB"):
        (tmp_path / "images" / sub).mkdir(parents=True)
        (tmp_path / "images" / sub / "same.jpg").write_bytes(b"x")
    (tmp_path / "t_test.json").write_text(_json.dumps([{
        "id": "0", "image": ["same.jpg"],
        "conversations": [{"from": "human", "value": "(A) (B)"},
                          {"from": "gpt", "value": "A"}]}]))
    items = load_task(tmp_path, "t")
    assert not items[0].images[0].exists(), "ambiguous basename must not resolve"


def test_ambiguity_broken_by_a_matching_path_tail(tmp_path):
    """'optA/same.jpg' is unambiguous even though 'same.jpg' alone is not."""
    import json as _json

    from gavagai.data.devcv import load_task

    for sub in ("optA", "optB"):
        (tmp_path / "images" / sub).mkdir(parents=True)
        (tmp_path / "images" / sub / "same.jpg").write_bytes(b"x")
    (tmp_path / "t_test.json").write_text(_json.dumps([{
        "id": "0", "image": ["optA/same.jpg"],
        "conversations": [{"from": "human", "value": "(A) (B)"},
                          {"from": "gpt", "value": "A"}]}]))
    items = load_task(tmp_path, "t")
    assert items[0].images[0].exists()
    assert items[0].images[0].parent.name == "optA"
