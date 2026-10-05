"""MD3 --- Margin Density Drift Detection, as a baseline.

Reimplementation of the label-free drift detector of Sethi & Kantardzic
(2017), *Expert Systems with Applications* 82:77-99, adapted to the
multi-class, lot-structured setting used here.

This baseline matters more than the others implemented here: MD3 was
introduced specifically to beat the posterior-probability tracking family
(Dries & Ruckert; Lindstrom et al.), which is exactly the KS / ADWIN / Page-Hinkley-on-confidence baselines used for comparison here. Their
Table 11 reports margin density reacting only after 11 relevant features drift
where posterior-uncertainty tracking fires after 6, i.e. MD3 is the more
false-alarm-robust of the two. Beating confidence tracking therefore says very
little; beating MD3 is the test that counts.

Two variants, mirroring the paper:

``MD3-conf``
    Margin density read off the *deployed* classifier. The paper defines the
    binary case as ``|p(y=+1|x) - p(y=-1|x)| < theta``; the multi-class
    generalisation used here is the gap between the two highest class
    probabilities, which reduces to their definition when D = 2.

``MD3-RS``
    Blindspot density from a random-subspace ensemble --- the paper's general
    form, applicable to classifiers with no explicit margin. Defaults follow
    the paper: 20 base trees, 50% of features each, uncertainty threshold 0.5.

Two deliberate departures from the paper, both in the baseline's favour or
neutral, and both stated so the comparison cannot be accused of hobbling it:

1. The paper signals when ``|MD_t - MD_ref| > theta * sigma_ref``, with
   ``sigma_ref`` from K-fold cross-validation on the training set. Here the
   *same* absolute-deviation statistic is exposed as a score and thresholded
   empirically on held-out in-control lots, exactly as every other detector in
   this study is. That replaces one tuning constant with the common
   calibration and makes the false-alarm rates comparable.
2. Smoothing uses the study-wide lambda rather than the paper's
   ``(N-1)/N``, so every EWMA-type detector here shares a smoothing constant.

The absolute value matters and is not an implementation detail: the paper
shows drift can *lower* margin density as well as raise it (their scenario
C0-C1), so a one-sided statistic misses half the cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


def margin_indicator(probabilities: np.ndarray, theta: float = 0.5) -> np.ndarray:
    """Per-sample indicator: is this sample inside the classifier's margin?

    The margin is the gap between the top two class probabilities. A small gap
    means the model is close to switching its answer, which is the multi-class
    reading of the paper's binary confidence.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape[1] < 2:
        raise ValueError("need at least two classes")
    top2 = np.partition(probabilities, -2, axis=1)[:, -2:]
    gap = top2[:, 1] - top2[:, 0]
    return (gap <= theta).astype(float)


@dataclass
class RandomSubspaceEnsemble:
    """Feature-bagged ensemble used for the blindspot-density variant."""

    model: BaggingClassifier
    classes: list[str]

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(features)
        out = np.zeros((len(features), len(self.classes)), dtype=float)
        for j, name in enumerate(self.model.classes_):
            out[:, self.classes.index(str(name))] = proba[:, j]
        totals = out.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return out / totals


def train_random_subspace(
    features: np.ndarray,
    labels,
    classes: list[str],
    n_estimators: int = 20,
    feature_fraction: float = 0.5,
    seed: int = 0,
) -> RandomSubspaceEnsemble:
    """Random-subspace ensemble of decision trees, per the paper's MD3-RS.

    The paper is explicit that the base learner must spread importance across
    features --- it shows MD3 failing on an L1-penalised model precisely
    because that model concentrates on few features. Unpruned trees over
    random half-subspaces satisfy that condition.
    """
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=seed),
        n_estimators=n_estimators,
        max_features=feature_fraction,
        bootstrap=False,
        bootstrap_features=False,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(features, labels)
    return RandomSubspaceEnsemble(model=model, classes=list(classes))


def lot_margin_density(
    indicator: np.ndarray,
    lots: np.ndarray,
    lot_sequence: list[str],
) -> np.ndarray:
    """Aggregate the per-wafer margin indicator to one density per lot."""
    lots = np.asarray(lots, dtype=object)
    index: dict[str, list[int]] = {}
    for position, lot in enumerate(lots):
        index.setdefault(str(lot), []).append(position)
    return np.asarray(
        [
            float(np.mean(indicator[index[str(lot)]]))
            for lot in lot_sequence
            if str(lot) in index
        ]
    )


def md3_score(
    density: np.ndarray,
    phase_one: int,
    lam: float = 0.2,
) -> np.ndarray:
    """Absolute EWMA deviation of margin density from its in-control level.

    Returned for lots ``[phase_one:]``, matching every other scorer in this
    study, so delays are measured from the same origin.
    """
    density = np.asarray(density, dtype=float)
    reference = float(density[:phase_one].mean())

    smoothed = np.empty(len(density))
    state = reference
    for t, value in enumerate(density):
        state = lam * value + (1.0 - lam) * state
        smoothed[t] = state
    return np.abs(smoothed[phase_one:] - reference)
