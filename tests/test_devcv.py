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


def test_load_and_parse():
    items = load_task(FIX, "picture_vocabulary", "test")
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
    items = load_task(FIX, "picture_vocabulary", "test")
    # An oracle scorer that always prefers the gold image must score 1.0.
    gold_path = items[0].images[1]
    res = evaluate_picture_vocabulary(items, lambda w, p: 1.0 if p == gold_path else 0.0)
    assert res["accuracy"] == 1.0 and res["n"] == 1 and res["chance"] == 0.25
    # A scorer that always prefers a distractor must score 0.0.
    bad = items[0].images[0]
    res = evaluate_picture_vocabulary(items, lambda w, p: 1.0 if p == bad else 0.0)
    assert res["accuracy"] == 0.0


def test_localization_scoring():
    items = load_task(FIX, "picture_vocabulary", "test")
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

    items = load_task(FIX, "leftright", "test")
    feats = [image_feature(p) for p in items[0].images]
    sims = [float(feats[0] @ f) for f in feats[1:]]
    assert int(max(range(3), key=sims.__getitem__)) == items[0].answer_index

    res = audit_task([(FIX / "test", "leftright", None)])
    assert res["match_acc"] == 1.0
