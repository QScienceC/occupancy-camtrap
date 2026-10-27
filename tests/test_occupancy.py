"""The occupancy models: analytic gradients match finite differences, and fits recover the
occupancy of simulated surveys."""

import numpy as np
import pytest

from occreview import occupancy


def simulate(rng, sites=300, occasions=8, psi=0.4, p=0.3, varying_days=True):
    z = rng.random(sites) < psi
    days = (
        rng.integers(3, 8, (sites, occasions)) if varying_days else np.full((sites, occasions), 7)
    )
    observed = rng.random((sites, occasions)) > 0.05
    y = z[:, None] & (rng.random((sites, occasions)) < p) & observed
    return occupancy.History(y, observed, days)


def numeric_gradient(nll, theta, h=1e-6):
    out = np.empty_like(theta)
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = h
        out[i] = (nll(theta + step)[0] - nll(theta - step)[0]) / (2 * h)
    return out


def test_occupancy_gradient():
    h = simulate(np.random.default_rng(1))
    nll = occupancy.occupancy_nll(h)
    theta = np.array([0.3, -1.0, 0.05])
    assert np.allclose(nll(theta)[1], numeric_gradient(nll, theta), rtol=1e-5, atol=1e-5)


def test_false_positive_gradient():
    rng = np.random.default_rng(2)
    h = simulate(rng)
    h.certain = h.y & (rng.random(h.y.shape) < 0.5)
    h.reviewed = rng.random(h.y.shape)
    nll = occupancy.false_positive_nll(h)
    theta = np.array([0.2, -1.2, 0.1, -2.5, -0.3, 1.1])
    assert np.allclose(nll(theta)[1], numeric_gradient(nll, theta), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_occupancy_recovers_psi(seed):
    h = simulate(np.random.default_rng(seed), sites=2000, varying_days=False)
    fit = occupancy.occupancy(h)
    assert fit.converged and not fit.at_limit
    assert abs(fit.value - 0.4) < 0.05
    assert fit.lower < 0.4 < fit.upper


def test_false_positive_model_recovers_psi():
    rng = np.random.default_rng(6)
    h = simulate(rng, sites=2000, varying_days=False)
    occupied = h.y.any(axis=1)
    # Suggestions alone add detections at unoccupied sites now and then; reviews confirm half.
    false = ~occupied[:, None] & (rng.random(h.y.shape) < 0.03) & h.observed
    h.certain = h.y & (rng.random(h.y.shape) < 0.5)
    h.y = h.y | false
    h.reviewed = np.full(h.y.shape, 0.5)
    fit = occupancy.false_positives(h)
    assert fit.converged
    assert abs(fit.value - 0.4) < 0.06


def test_a_species_seen_everywhere_is_at_its_limit():
    rng = np.random.default_rng(7)
    h = simulate(rng, sites=50, psi=1.0, p=0.9, varying_days=False)
    h.y[:, 0] = True
    fit = occupancy.occupancy(h)
    assert fit.at_limit and fit.lower is None


def scored(evidence_of, seed=3):
    """A simulated survey with every third observed occasion left to scores: their evidence is
    evidence_of(detected) from what the simulation says there."""
    h = simulate(np.random.default_rng(seed))
    left = h.observed & (np.indices(h.y.shape).sum(axis=0) % 3 == 0)
    h.evidence = np.where(left, evidence_of(h.y), np.nan)
    return h, left


def test_score_gradient():
    rng = np.random.default_rng(4)
    h, _ = scored(lambda y: np.where(y, 3.0, -3.0) + rng.normal(0, 1, y.shape))
    h.y = h.y & np.isnan(h.evidence)
    nll = occupancy.score_nll(h)
    theta = np.array([0.3, -1.0, 0.05])
    assert np.allclose(nll(theta)[1], numeric_gradient(nll, theta), rtol=1e-5, atol=1e-5)


def test_scores_at_their_extremes_count_as_reviews_would():
    """Overwhelming evidence counts as a detection and evidence against as none, so the fit is
    occupancy's on the history; no evidence is an occasion no camera ran."""
    plain = simulate(np.random.default_rng(3))
    sure, left = scored(lambda y: np.where(y, 40.0, -40.0))
    sure.y = sure.y & ~left
    assert occupancy.score_occupancy(sure).value == pytest.approx(
        occupancy.occupancy(plain).value, abs=1e-4
    )
    blank, left = scored(lambda y: np.zeros(y.shape))
    blank.y = blank.y & ~left
    unseen = simulate(np.random.default_rng(3))
    unseen.observed = unseen.observed & ~left
    unseen.y = unseen.y & unseen.observed
    assert occupancy.score_occupancy(blank).value == pytest.approx(
        occupancy.occupancy(unseen).value, abs=1e-4
    )
