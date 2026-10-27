"""The project model: a classifier of the survey's own species, trained on the fingerprints of the
reviewed boxes, as WildObserve trains it: the fingerprints standardized, then a multinomial
logistic regression with balanced class weights and C = 0.5 (scikit-learn's StandardScaler and
LogisticRegression). A species needs 3 labeled boxes to be learned, and the model 2 species.

Its suggestion for a box is its top species, with that probability as the score. The lead of the
top suggestion over the runner-up (none when it disagrees with SpeciesNet) orders WildObserve's
Close calls list. The two agree when SpeciesNet's top label is the species' name, or a label that
SpeciesNet gave as its top guess on at least a quarter of that species' training boxes.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MIN_EXAMPLES = 3
MIN_SPECIES = 2
LABEL_SHARE = 0.25
MIN_SCORE = 0.01  # a runner-up under this isn't offered (so the lead is the top score itself)


@dataclass
class ProjectModel:
    species: list[str]
    scaler: StandardScaler
    model: LogisticRegression
    speciesnet_labels: dict[str, set[str]]
    examples: int

    def probabilities(self, fingerprints: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(fingerprints.astype(np.float32))
        p = self.model.predict_proba(z)
        return p

    def judge(self, fingerprints: np.ndarray, speciesnet_label: np.ndarray):
        """For every box: the top species, its score (rounded as WildObserve stores it), and the
        lead over the runner-up, 0 when SpeciesNet disagrees."""
        p = self.probabilities(fingerprints)
        order = np.argsort(-p, axis=1)
        top, second = order[:, 0], order[:, 1]
        score = np.round(p[np.arange(len(p)), top], 4)
        runner_up = np.round(p[np.arange(len(p)), second], 4)
        runner_up = np.where(runner_up >= MIN_SCORE, runner_up, 0.0)
        names = [self.species[i] for i in top]
        agrees = np.array(
            [
                name.lower() == str(label).lower() or str(label) in self.speciesnet_labels[name]
                for name, label in zip(names, speciesnet_label, strict=True)
            ]
        )
        margin = np.where(agrees, np.round(score - runner_up, 4), 0.0)
        return names, score, margin, agrees


def train(fingerprints: np.ndarray, names: list[str], speciesnet_label: np.ndarray):
    """A project model from labeled boxes, or None if too few species have enough of them."""
    counts = Counter(names)
    keep = np.array([counts[n] >= MIN_EXAMPLES for n in names])
    learnable = sorted({n for n, k in zip(names, keep, strict=True) if k})
    if len(learnable) < MIN_SPECIES:
        return None
    x = fingerprints[keep].astype(np.float64)
    y = [n for n, k in zip(names, keep, strict=True) if k]
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(scaler.transform(x), y)
    labels = speciesnet_label[keep]
    usual: dict[str, set[str]] = {}
    for name in learnable:
        mine = Counter(str(label) for label, n in zip(labels, y, strict=True) if n == name)
        total = sum(mine.values())
        usual[name] = {label for label, k in mine.items() if k / total >= LABEL_SHARE}
    return ProjectModel(list(model.classes_), scaler, model, usual, len(y))
