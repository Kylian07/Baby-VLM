"""Tests for the GAVAGAI objective."""
import torch
import torch.nn.functional as F

from gavagai.losses import gavagai_loss, infonce_loss



def _batch(b=4, n=6, m=5, d=32):
    w = F.normalize(torch.randn(b, n, d), dim=-1).requires_grad_()
    v = F.normalize(torch.randn(b, m, d), dim=-1)
    return w, v, torch.ones(b, n, dtype=torch.bool), torch.ones(b, m, dtype=torch.bool)


def test_loss_is_finite_and_differentiable_in_all_modes():
    torch.manual_seed(0)
    w, v, wm, sm = _batch()
    for rho in (0.0, 0.5, 2.0, None):
        for use_null in (True, False):
            w.grad = None
            loss, stats = gavagai_loss(w, v, wm, sm, rho=rho, use_null=use_null)
            assert torch.isfinite(loss), (rho, use_null)
            loss.backward(retain_graph=True)
            assert torch.isfinite(w.grad).all() and w.grad.abs().sum() > 0


def test_without_null_every_word_is_referential():
    torch.manual_seed(0)
    w, v, wm, sm = _batch()
    _, stats = gavagai_loss(w, v, wm, sm, rho=1.0, use_null=False)
    assert abs(float(stats["referential_mass"]) - 1.0) < 1e-5


def test_null_bin_lets_words_opt_out():
    """With a high null threshold, most mass should leave the real slots."""
    torch.manual_seed(0)
    w, v, wm, sm = _batch()
    _, low = gavagai_loss(w, v, wm, sm, rho=1.0, use_null=True, kappa=-1.0)
    _, high = gavagai_loss(w, v, wm, sm, rho=1.0, use_null=True, kappa=1.0)
    assert float(high["referential_mass"]) < float(low["referential_mass"])


def test_exclusivity_reduces_slot_concentration():
    """The mechanism of Proposition 2, measured directly."""
    torch.manual_seed(0)
    w, v, wm, sm = _batch(b=8, n=10, m=6)
    ginis = [
        float(gavagai_loss(w, v, wm, sm, rho=r, use_null=False)[1]["slot_usage_gini"])
        for r in (0.0, 0.1, 1.0, None)
    ]
    assert ginis[0] > ginis[-1]
    assert all(ginis[i] >= ginis[i + 1] - 1e-6 for i in range(len(ginis) - 1)), ginis


def test_loss_ignores_padded_words_and_slots():
    """Changing padded entries must not change the loss.

    A silent failure here would make every variable-length batch subtly wrong.
    """
    torch.manual_seed(0)
    b, n, m, d = 3, 7, 6, 32
    w = F.normalize(torch.randn(b, n, d), dim=-1)
    v = F.normalize(torch.randn(b, m, d), dim=-1)
    wm = torch.ones(b, n, dtype=torch.bool)
    sm = torch.ones(b, m, dtype=torch.bool)
    wm[0, 5:] = False
    sm[1, 4:] = False

    base = gavagai_loss(w, v, wm, sm, rho=1.0)[0]

    w2, v2 = w.clone(), v.clone()
    w2[0, 5:] = F.normalize(torch.randn(2, d), dim=-1)
    v2[1, 4:] = F.normalize(torch.randn(2, d), dim=-1)
    perturbed = gavagai_loss(w2, v2, wm, sm, rho=1.0)[0]

    assert torch.allclose(base, perturbed, atol=1e-5), (base.item(), perturbed.item())


def test_infonce_baseline_is_minimised_by_matched_pairs():
    torch.manual_seed(0)
    x = F.normalize(torch.randn(16, 32), dim=-1)
    aligned = infonce_loss(x, x)
    shuffled = infonce_loss(x, x[torch.randperm(16)])
    assert aligned < shuffled
