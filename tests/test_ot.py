"""Correctness tests for the OT core, including the two claims the paper rests on."""
import torch

from gavagai.ot import NEG_INF, referential_plan, sinkhorn_log, sinkhorn_scale



def test_row_marginal_is_exact():
    torch.manual_seed(0)
    C = torch.randn(3, 5, 7)
    log_a = torch.log(torch.full((3, 5), 1 / 5))
    log_b = torch.log(torch.full((3, 7), 1 / 7))
    P = sinkhorn_log(C, log_a, log_b, eps=0.05, rho=None, n_iter=300).exp()
    assert torch.allclose(P.sum(-1), torch.full((3, 5), 1 / 5), atol=1e-6)


def test_balanced_recovers_column_marginal():
    torch.manual_seed(0)
    C = torch.randn(2, 6, 6)
    log_a = torch.log(torch.full((2, 6), 1 / 6))
    log_b = torch.log(torch.full((2, 6), 1 / 6))
    P = sinkhorn_log(C, log_a, log_b, eps=0.05, rho=None, n_iter=500).exp()
    assert torch.allclose(P.sum(-2), torch.full((2, 6), 1 / 6), atol=1e-4)


def test_rho_zero_is_exactly_row_softmax():
    """rho = 0 must reproduce the InfoNCE-style row-wise softmax bit for bit."""
    torch.manual_seed(0)
    C = torch.randn(4, 5, 9)
    eps = 0.13
    log_a = torch.log(torch.full((4, 5), 1 / 5))
    log_b = torch.log(torch.full((4, 9), 1 / 9))
    P = sinkhorn_log(C, log_a, log_b, eps=eps, rho=0.0, n_iter=50).exp()
    expected = torch.softmax(-C / eps, dim=-1) / 5
    assert torch.allclose(P, expected, atol=1e-6)


def test_masked_rows_carry_no_mass():
    torch.manual_seed(0)
    C = torch.randn(2, 4, 6)
    log_a = torch.log(torch.tensor([[0.5, 0.5, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3, 0.0]]).clamp_min(1e-30))
    log_a = torch.where(log_a < -60, torch.full_like(log_a, NEG_INF), log_a)
    log_b = torch.log(torch.full((2, 6), 1 / 6))
    P = sinkhorn_log(C, log_a, log_b, eps=0.05, rho=1.0, n_iter=200).exp()
    assert P[0, 2:].abs().max() < 1e-12
    assert P[1, 3].abs().max() < 1e-12


def test_rho_interpolates_monotonically():
    """Larger rho => column marginals closer to the target (more exclusivity)."""
    torch.manual_seed(0)
    C = torch.randn(1, 8, 8)
    log_a = torch.log(torch.full((1, 8), 1 / 8))
    log_b = torch.log(torch.full((1, 8), 1 / 8))
    errs = []
    for rho in [0.0, 0.05, 0.2, 1.0, 10.0]:
        P = sinkhorn_log(C, log_a, log_b, eps=0.1, rho=rho, n_iter=500).exp()
        errs.append((P.sum(-2) - 1 / 8).abs().sum().item())
    assert all(errs[i] >= errs[i + 1] - 1e-9 for i in range(len(errs) - 1)), errs


def test_null_bin_absorbs_unmatched_words():
    """A word with no good slot should be judged non-referential."""
    torch.manual_seed(0)
    sim = torch.tensor([[[0.9, 0.1, 0.0], [-0.8, -0.7, -0.9]]])  # word 0 matches, word 1 does not
    plan, referential = referential_plan(sim, kappa=0.0, eps=0.05, rho=1.0, null_prior=0.5)
    assert referential[0, 0] > 0.9
    assert referential[0, 1] < 0.1


# ---------------------------------------------------------------------------
# Proposition 1: frequency bias of row-softmax vs. unbiasedness of balanced OT
# ---------------------------------------------------------------------------

def _lexicon_setup():
    """A 3-word / 3-concept world where concept 0 is 20x more frequent.

    Word w is emitted by its true referent c(w)=w with prob 0.6 and by any
    concept with prob 0.2 (noise).  Concept prior pi is heavily skewed.
    """
    pi = torch.tensor([0.90, 0.05, 0.05])          # concept prior
    lik = torch.tensor([                            # p(w | k), rows = concepts
        [0.60, 0.20, 0.20],
        [0.20, 0.60, 0.20],
        [0.20, 0.20, 0.60],
    ])
    joint = pi.unsqueeze(1) * lik                   # p(k, w), (K, V)
    return pi, lik, joint


def test_row_softmax_is_frequency_biased():
    """argmax_k p(k|w) collapses onto the frequent concept -- the hubness failure."""
    torch.manual_seed(0)
    _, _, joint = _lexicon_setup()
    posterior = joint / joint.sum(0, keepdim=True)  # p(k|w)
    predicted = posterior.argmax(0)
    assert (predicted == torch.tensor([0, 0, 0])).all(), predicted


def test_balanced_ot_recovers_true_lexicon():
    """Sinkhorn scaling ranks by PMI, whose argmax is argmax_k p(w|k) -- unbiased."""
    torch.manual_seed(0)
    _, lik, joint = _lexicon_setup()
    scaled = sinkhorn_scale(joint)
    predicted = scaled.argmax(0)
    assert (predicted == torch.tensor([0, 1, 2])).all(), predicted
    # and it agrees with the likelihood argmax, as the proposition claims
    assert (predicted == lik.argmax(0)).all()


def test_sinkhorn_scaling_log_equals_pmi_up_to_constants():
    """log(D_r J D_c) = PMI(J) + u_i + v_j for some vectors u, v."""
    torch.manual_seed(0)
    _, _, joint = _lexicon_setup()
    joint = joint.double()
    scaled = sinkhorn_scale(joint)
    pmi = torch.log(joint / (joint.sum(1, keepdim=True) * joint.sum(0, keepdim=True)))
    diff = torch.log(scaled) - pmi
    # a matrix is of the form u_i + v_j iff all 2x2 "second differences" vanish
    second = diff[1:, 1:] - diff[:1, 1:] - diff[1:, :1] + diff[:1, :1]
    assert second.abs().max() < 1e-10, second
