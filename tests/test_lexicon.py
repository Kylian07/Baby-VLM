"""Tests for the persistent cross-situational lexicon."""
import torch
import torch.nn.functional as F

from gavagai.lexicon import CrossSituationalLexicon



def _lex(vocab=40, k=16, d=24, **kw):
    return CrossSituationalLexicon(vocab_size=vocab, n_prototypes=k, dim=d, **kw)


def test_assignments_are_simplices_and_respect_the_mask():
    torch.manual_seed(0)
    lex = _lex()
    slots = F.normalize(torch.randn(4, 5, 24), dim=-1)
    mask = torch.ones(4, 5, dtype=torch.bool)
    mask[0, 3:] = False
    q = lex.assign(slots, mask)
    assert q.shape == (4, 5, 16)
    assert (q >= 0).all()
    real = q[mask]
    assert torch.allclose(real.sum(-1), torch.ones(len(real)), atol=1e-4)
    assert q[0, 3:].sum() < 1e-6


def test_balanced_assignment_spreads_over_prototypes():
    """The Sinkhorn step exists to stop the codebook collapsing onto a few codes."""
    torch.manual_seed(0)
    lex = _lex(vocab=10, k=32, d=24)
    slots = F.normalize(torch.randn(16, 8, 24), dim=-1)
    mask = torch.ones(16, 8, dtype=torch.bool)
    balanced = lex.assign(slots, mask).reshape(-1, 32).sum(0)
    plain = torch.softmax(
        slots.reshape(-1, 24) @ lex.prototypes.t() / lex.assign_temp, dim=-1
    ).sum(0)
    # Coefficient of variation: lower means more evenly used.
    cv = lambda x: (x.std() / x.mean()).item()
    assert cv(balanced) < cv(plain), (cv(balanced), cv(plain))


def test_bonus_is_withheld_during_warmup():
    torch.manual_seed(0)
    lex = _lex()
    slots = F.normalize(torch.randn(2, 4, 24), dim=-1)
    mask = torch.ones(2, 4, dtype=torch.bool)
    q = lex.assign(slots, mask)
    ids = torch.randint(0, 40, (2, 3))
    assert lex.bonus(ids, q, warmup=10) is None
    lex.n_updates += 50
    b = lex.bonus(ids, q, warmup=10)
    assert b.shape == (2, 3, 4) and torch.isfinite(b).all()


def test_pmi_recovers_a_planted_association_despite_a_frequent_distractor():
    """Proposition 3 at the module level.

    Word 0 always co-occurs with concept 1.  Concept 0 is a background that
    co-occurs with *everything* far more often.  Raw counts point at the
    background; PMI points at the true referent.
    """
    torch.manual_seed(0)
    lex = _lex(vocab=5, k=4, d=8, smoothing=1e-3)
    counts = torch.zeros(5, 4)
    counts[:, 0] = 100.0        # background concept, present for every word
    for w in range(5):
        counts[w, (w % 3) + 1] = 10.0   # each word's true referent
    lex.counts.copy_(counts)

    assert int(counts[0].argmax()) == 0, "raw counts point at the background"
    assert int(lex.pmi()[0].argmax()) == 1, "PMI points at the true referent"
    for w in range(5):
        assert int(lex.pmi()[w].argmax()) == (w % 3) + 1


def test_known_words_requires_both_evidence_and_a_peak():
    torch.manual_seed(0)
    lex = _lex(vocab=6, k=4, d=8, smoothing=1e-3)
    counts = torch.zeros(6, 4)
    counts[0, 1] = 50.0                       # strong, peaked
    counts[1, :] = 25.0                       # plenty of evidence, no peak
    counts[2, 1] = 1.0                        # peaked but barely observed
    lex.counts.copy_(counts)
    # With K=4 prototypes the largest reachable PMI is log(4) = 1.386, and the
    # observed peaks here are ~0.69, so the threshold has to sit below that.
    known = lex.known_words(threshold=0.5, min_count=5.0)
    assert bool(known[0]) and not bool(known[1]) and not bool(known[2])


def test_update_accumulates_alignment_mass_and_moves_prototypes():
    torch.manual_seed(0)
    lex = _lex(vocab=8, k=6, d=12, proto_momentum=0.5)
    slots = F.normalize(torch.randn(3, 4, 12), dim=-1)
    smask = torch.ones(3, 4, dtype=torch.bool)
    q = lex.assign(slots, smask)
    ids = torch.randint(0, 8, (3, 5))
    plan = torch.rand(3, 5, 4)
    before = lex.prototypes.clone()
    lex.update(ids, q, plan, torch.ones(3, 5, dtype=torch.bool), slot_emb=slots)
    assert lex.counts.sum() > 0
    assert int(lex.n_updates) == 1
    assert not torch.allclose(before, lex.prototypes)
    assert torch.allclose(lex.prototypes.norm(dim=-1), torch.ones(6), atol=1e-5)
