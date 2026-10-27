"""Single-season occupancy models, fitted by maximum likelihood, as the R package unmarked fits
them (and as WildObserve does):

- occupancy (MacKenzie et al. 2002; unmarked's occu(~ days ~ 1)): a site is occupied with
  probability psi; at an occupied site the species is detected in an occasion with probability p,
  on the logit scale linear in the occasion's camera-days when they vary;
- occupancy allowing for false positives (Miller et al. 2011; unmarked's occuFP with type 3 data):
  an occasion's outcome is 0 (not detected), 1 (detected by suggestions alone) or 2 (detected in a
  reviewed image). At an occupied site the species is detected with probability p, and a detection
  is confirmed with probability b, on the logit scale linear in the share of the occasion's animal
  images reviewed (when that varies where the species was detected); at an unoccupied site nothing
  is confirmed, and suggestions alone detect it with probability f;
- score-based occupancy (a continuous-score model, Rhinehart et al. 2022, with the scores'
  calibration fitted apart, as Haucke et al. 2026 fit it; scores.py): occupancy as above, where
  an occasion that reviews don't settle weighs its classifier score as evidence, a log Bayes
  factor, instead of being detected or not (WildObserve's score-based option).

The negative log-likelihood is minimized by L-BFGS (SciPy) with its analytic gradient; standard
errors come from the inverse of the Hessian, taken by central differences of the gradient; a 95%
interval for occupancy is made on the logit scale and transformed back. A fit has converged when
no component of the gradient exceeds 1e-3 at its end, and occupancy is at its limit when |logit
psi| exceeds 8 (psi beyond 0.0003 or 0.9997).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

Z = 1.959963984540054  # a 95% interval's half-width, in standard errors
CONVERGED = 1e-3
LIMIT = 8.0
WIDEST = 10.0  # a standard error on the logit scale past this: no interval


def logistic(x):
    return 0.5 * (1 + np.tanh(np.asarray(x) / 2))


def log1pexp(x):
    return np.logaddexp(0, x)


@dataclass
class History:
    y: np.ndarray  # sites by occasions: detected
    observed: np.ndarray  # a camera ran
    days: np.ndarray  # camera-days
    certain: np.ndarray | None = None  # a reviewed image confirms the detection
    reviewed: np.ndarray | None = None  # share of the occasion's animal images reviewed
    evidence: np.ndarray | None = None  # score-based: a log Bayes factor where scores decide

    def detection_design(self) -> np.ndarray:
        """An intercept, and the camera-days when they vary among the occasions observed."""
        ones = np.ones(self.days.shape)
        if np.unique(self.days[self.observed]).size > 1:
            return np.stack([ones, self.days.astype(float)], axis=-1)
        return ones[..., None]

    def review_design(self) -> np.ndarray:
        """An intercept, and the reviewed share when it varies where the species was detected."""
        ones = np.ones(self.reviewed.shape)
        detected = self.y & self.observed
        if np.unique(self.reviewed[detected]).size > 1:
            return np.stack([ones, self.reviewed], axis=-1)
        return ones[..., None]


@dataclass
class Estimate:
    value: float
    lower: float | None
    upper: float | None
    converged: bool
    at_limit: bool


def occupancy_nll(h: History):
    y, m = h.y.astype(float), h.observed
    detected = (h.y & m).any(axis=1)
    design = h.detection_design()

    def nll(theta):
        beta, alpha = theta[0], theta[1:]
        mu = design @ alpha
        p = logistic(mu)
        # log P(history | occupied) at each site
        log_h = np.where(m, y * mu - log1pexp(mu), 0).sum(axis=1)
        log_psi, log_not = -log1pexp(-beta), -log1pexp(beta)
        log_l = np.where(detected, log_psi + log_h, np.logaddexp(log_psi + log_h, log_not))
        occupied = np.exp(log_psi + log_h - log_l)  # P(occupied | history)
        d_beta = (occupied - logistic(beta)).sum()
        d_alpha = np.einsum("i,ijk->k", occupied, np.where(m, y - p, 0)[..., None] * design)
        return -log_l.sum(), -np.concatenate([[d_beta], d_alpha])

    return nll


def false_positive_nll(h: History):
    m = h.observed
    detected = h.y & m
    certain = detected & h.certain
    confirmed = certain.any(axis=1)
    y = detected.astype(float)
    design, review = h.detection_design(), h.review_design()
    k = design.shape[-1]

    def nll(theta):
        beta, alpha, kappa, gamma = theta[0], theta[1 : 1 + k], theta[1 + k], theta[2 + k :]
        mu, nu = design @ alpha, review @ gamma
        p, b, f = logistic(mu), logistic(nu), logistic(kappa)
        log_h = np.where(m, y * mu - log1pexp(mu), 0).sum(axis=1) + np.where(
            detected, np.where(certain, -log1pexp(-nu), -log1pexp(nu)), 0
        ).sum(axis=1)
        log_unoccupied = np.where(m, y * kappa - log1pexp(kappa), 0).sum(axis=1)
        log_psi, log_not = -log1pexp(-beta), -log1pexp(beta)
        log_l = np.where(
            confirmed,
            log_psi + log_h,
            np.logaddexp(log_psi + log_h, log_not + log_unoccupied),
        )
        occupied = np.exp(log_psi + log_h - log_l)
        d_beta = (occupied - logistic(beta)).sum()
        d_alpha = np.einsum("i,ijk->k", occupied, np.where(m, y - p, 0)[..., None] * design)
        d_kappa = ((1 - occupied) * np.where(m, y - f, 0).sum(axis=1)).sum()
        d_gamma = np.einsum(
            "i,ijk->k", occupied, np.where(detected, certain - b, 0)[..., None] * review
        )
        gradient = np.concatenate([[d_beta], d_alpha, [d_kappa], d_gamma])
        return -log_l.sum(), -gradient

    return nll


def score_nll(h: History):
    """Score-based occupancy: an occasion that reviews settle (evidence NaN) counts as in
    occupancy_nll; one whose answer rests on a score with log Bayes factor l adds, at an occupied
    site, log((1 - p) + p e^l), and nothing at an unoccupied one (the score's density without
    the species divides out)."""
    m = h.observed
    scored = m & np.isfinite(h.evidence)
    settled = m & ~scored
    ell = np.where(scored, h.evidence, 0.0)
    y = (h.y & settled).astype(float)
    detected = (h.y & settled).any(axis=1)
    design = h.detection_design()

    def nll(theta):
        beta, alpha = theta[0], theta[1:]
        mu = design @ alpha
        p = logistic(mu)
        log_h = (
            np.where(settled, y * mu - log1pexp(mu), 0)
            + np.where(scored, log1pexp(mu + ell) - log1pexp(mu), 0)
        ).sum(axis=1)
        log_psi, log_not = -log1pexp(-beta), -log1pexp(beta)
        log_l = np.where(detected, log_psi + log_h, np.logaddexp(log_psi + log_h, log_not))
        occupied = np.exp(log_psi + log_h - log_l)
        d_beta = (occupied - logistic(beta)).sum()
        d_mu = np.where(settled, y - p, 0) + np.where(scored, logistic(mu + ell) - p, 0)
        d_alpha = np.einsum("i,ijk->k", occupied, d_mu[..., None] * design)
        return -log_l.sum(), -np.concatenate([[d_beta], d_alpha])

    return nll


def hessian(nll, theta: np.ndarray) -> np.ndarray:
    size = theta.size
    found = np.empty((size, size))
    for i in range(size):
        step = np.zeros(size)
        step[i] = 1e-5 * max(1.0, abs(theta[i]))
        found[i] = (nll(theta + step)[1] - nll(theta - step)[1]) / (2 * step[i])
    return (found + found.T) / 2


def fit(nll, start: np.ndarray) -> Estimate:
    """Occupancy (the first parameter, on the logit scale) and its interval."""
    found = minimize(
        nll,
        start,
        jac=True,
        method="L-BFGS-B",
        # Stop when no gradient component exceeds 1e-4, as WildObserve's optimizer does (and
        # not on a small relative change in the likelihood, SciPy's default, which on large
        # surveys stops before the gradient is small). Going further only lets detection run
        # off toward a step in camera-days where a species' detections allow it.
        options={"maxiter": 10000, "gtol": 1e-4, "ftol": 1e-15, "maxls": 50},
    )
    theta = found.x
    value, gradient = nll(theta)
    converged = bool(np.isfinite(value) and np.abs(gradient).max() <= CONVERGED)
    psi = float(logistic(theta[0]))
    at_limit = bool(abs(theta[0]) > LIMIT)
    lower = upper = None
    if not at_limit and 0 < psi < 1:
        try:
            inverse = np.linalg.inv(hessian(nll, theta))
            covariance = (inverse + inverse.T) / 2
            if np.all(np.isfinite(covariance)) and np.all(np.linalg.eigvalsh(covariance) > 0):
                se = float(np.sqrt(covariance[0, 0]))
                if se <= WIDEST:
                    lower = float(logistic(theta[0] - Z * se))
                    upper = float(logistic(theta[0] + Z * se))
        except np.linalg.LinAlgError:
            pass
    return Estimate(psi, lower, upper, converged, at_limit)


def detection_start(h: History) -> float:
    detected = (h.y & h.observed).any(axis=1)
    occasions = h.observed[detected].sum()
    rate = np.clip(h.y[detected].sum() / occasions if occasions else 0.5, 0.05, 0.95)
    return float(np.log(rate / (1 - rate)))


def occupancy(h: History) -> Estimate:
    psi = np.clip((h.y & h.observed).any(axis=1).mean(), 0.05, 0.95)
    start = np.zeros(1 + h.detection_design().shape[-1])
    start[0], start[1] = np.log(psi / (1 - psi)), detection_start(h)
    return fit(occupancy_nll(h), start)


def false_positives(h: History) -> Estimate:
    confirmed = (h.y & h.observed & h.certain).any(axis=1).mean()
    psi = np.clip(max(confirmed, 0.05), 0.05, 0.95)
    k = h.detection_design().shape[-1]
    start = np.zeros(1 + k + 1 + h.review_design().shape[-1])
    start[0], start[1], start[1 + k] = np.log(psi / (1 - psi)), detection_start(h), -3.0
    return fit(false_positive_nll(h), start)


def score_occupancy(h: History) -> Estimate:
    likely = (h.y & h.observed) | (np.nan_to_num(h.evidence) > 0)
    psi = np.clip(likely.any(axis=1).mean(), 0.05, 0.95)
    start = np.zeros(1 + h.detection_design().shape[-1])
    start[0], start[1] = np.log(psi / (1 - psi)), detection_start(h)
    return fit(score_nll(h), start)
