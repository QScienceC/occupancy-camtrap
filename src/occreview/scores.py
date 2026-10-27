"""Score-based occupancy's evidence, as WildObserve's score-based option makes it (its
occupancy_scores.py): instead of counting a suggestion over the cut-off as a detection, an occasion
that reviews don't settle weighs the scores of its boxes as evidence (Rhinehart et al. 2022), with
the scores' calibration fitted apart from the occupancy model (Haucke et al. 2026).

For each species at each site in each occasion: a reviewed image shows it, a certain detection;
otherwise, if an image not reviewed yet has a box with a suggestion (WildObserve scores the boxes it
fingerprinted), the highest score any such box has for the species is evidence; otherwise, not
detected. A score is the species' log-odds from the project model. How an occasion's highest score
falls with and without the species is two Normal distributions, fitted by EM: the occasions reviews
settle as known, the rest as a mixture. The project model learned from the reviewed boxes, so its
scores on them are surer than on the rest: the fit is the unsettled occasions' alone while they
still hold MIN_CONFIRMED or more likely detections and its distribution with the species lies nearer
the reviewed ones; otherwise the settled occasions join it. An occasion's evidence is its score's
log Bayes factor, held past either distribution's mean at least as strong as at that mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_CONFIRMED = 3  # occasions a reviewed image shows the species in, to calibrate its scores
SMALLEST_SD = 0.05
STEPS = 500
SETTLED = 1e-9  # EM stops once the log-likelihood gains less than this per score


def log_normal(x, mean: float, sd: float):
    return -0.5 * ((x - mean) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)


@dataclass(frozen=True)
class Calibration:
    absent: tuple[float, float]  # mean and standard deviation of an occasion's highest score
    present: tuple[float, float]
    share: float  # of the unsettled occasions, the share with the species

    def evidence(self, scores: np.ndarray) -> np.ndarray:
        def ratio(x):
            return log_normal(x, *self.present) - log_normal(x, *self.absent)

        low, high = self.absent[0], self.present[0]
        found = ratio(scores)
        found = np.where(scores > high, np.maximum(found, ratio(high)), found)
        return np.where(scores < low, np.minimum(found, ratio(low)), found)


def calibrate(present, absent, unknown, fit_known: bool = True) -> Calibration | None:
    """Two Normal distributions by EM; without `fit_known`, the known scores only start it, from
    the unknown's own two groups (2-means). None with too few known detections, or when the
    distribution with the species doesn't lie above the other."""
    if len(present) < MIN_CONFIRMED:
        return None
    rest = absent if len(absent) >= 2 else unknown[unknown < present.mean()]
    if len(rest) < 2:
        return None
    mean = np.array([rest.mean(), present.mean()])
    sd = np.maximum([rest.std(), present.std()], SMALLEST_SD)
    share = 0.1
    if not fit_known:
        split = mean.mean()
        for _ in range(50):
            low, high = unknown[unknown <= split], unknown[unknown > split]
            if len(low) < 2 or len(high) < 2:
                break
            split, last = (low.mean() + high.mean()) / 2, split
            if abs(split - last) < 1e-6:
                break
        low, high = unknown[unknown <= split], unknown[unknown > split]
        if len(low) >= 2 and len(high) >= 2:
            mean = np.array([low.mean(), high.mean()])
            sd = np.maximum([low.std(), high.std()], SMALLEST_SD)
            share = len(high) / len(unknown)
    known = (absent, present) if fit_known else (absent[:0], present[:0])
    previous = -np.inf
    for _ in range(STEPS):
        there = np.log(share) + log_normal(unknown, mean[1], sd[1])
        not_there = np.log1p(-share) + log_normal(unknown, mean[0], sd[0])
        mixed = np.logaddexp(there, not_there)
        r = np.exp(there - mixed)
        total = mixed.sum()
        if fit_known:
            total += log_normal(present, mean[1], sd[1]).sum()
            total += log_normal(absent, mean[0], sd[0]).sum()
        if total - previous < SETTLED * (len(present) + len(absent) + len(unknown)):
            break
        previous = total
        for k, weight in ((0, 1 - r), (1, r)):
            n = len(known[k]) + weight.sum()
            if n < 1e-9:
                continue
            mean[k] = (known[k].sum() + weight @ unknown) / n
            spread = ((known[k] - mean[k]) ** 2).sum() + weight @ (unknown - mean[k]) ** 2
            sd[k] = max(np.sqrt(spread / n), SMALLEST_SD)
        share = float(np.clip(r.mean(), 1e-6, 1 - 1e-6)) if len(unknown) else 0.5
    if not mean[1] > mean[0]:
        return None
    return Calibration((float(mean[0]), float(sd[0])), (float(mean[1]), float(sd[1])), share)


def calibration_for(present, absent, unknown, known: bool | None = None) -> Calibration | None:
    """The project model's calibration (module docstring); `known` forces the settled occasions
    into the fit (True) or out of it (False)."""
    if known is not None:
        return calibrate(present, absent, unknown, fit_known=known)
    found = calibrate(present, absent, unknown, fit_known=False)
    if (
        found is not None
        and found.share * len(unknown) >= MIN_CONFIRMED
        and found.present[0] >= (found.absent[0] + present.mean()) / 2
    ):
        return found
    return calibrate(present, absent, unknown)


def log_odds(p: np.ndarray) -> np.ndarray:
    """Probabilities as log-odds, 1 - p kept off 0 (as WildObserve keeps it)."""
    return np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(1 - p, 1e-12))


def evidence(state, known: bool | None = None) -> dict[str, np.ndarray]:
    """Each species' evidence (sites by occasions; NaN where reviews settle an occasion), for the
    species the project model knows and has enough reviewed detections of to calibrate."""
    s = state.survey
    if state.model is None:
        return {}
    shape = s.effort.shape
    rows = s.box_image
    valid = s.occasion[rows] >= 0
    reviewed = state.reviewed[rows]
    usable = valid & (reviewed | s.suggestable)  # reviewed boxes, and those with a suggestion
    open_ = usable & ~reviewed
    i, j = s.site[rows], s.occasion[rows]
    scores = log_odds(state.model.probabilities(s.fingerprint))
    unsettled = np.zeros(shape, dtype=bool)
    unsettled[i[open_], j[open_]] = True
    observed = s.effort > 0
    seen = {}
    for row in np.flatnonzero(state.reviewed & (s.occasion >= 0)):
        for name in s.animals[row]:
            seen.setdefault(name, np.zeros(shape, dtype=bool))[s.site[row], s.occasion[row]] = True
    found: dict[str, np.ndarray] = {}
    for k, name in enumerate(state.model.species):
        best_any = np.full(shape, -np.inf)
        best_open = np.full(shape, -np.inf)
        np.maximum.at(best_any, (i[usable], j[usable]), scores[usable, k])
        np.maximum.at(best_open, (i[open_], j[open_]), scores[open_, k])
        confirmed = seen.get(name, np.zeros(shape, dtype=bool))
        scored = np.isfinite(best_any)
        present = best_any[confirmed & scored]
        absent = best_any[observed & ~confirmed & ~unsettled & scored]
        weighed = observed & ~confirmed & unsettled
        calibration = calibration_for(present, absent, best_open[weighed], known)
        if calibration is None:
            continue
        found[name] = np.full(shape, np.nan)
        found[name][weighed] = calibration.evidence(best_open[weighed])
    return found
