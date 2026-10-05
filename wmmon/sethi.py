"""Drift induction after Sethi & Kantardzic (2017).

Every degradation mode in `drift.py` is ours. One of them --- defect remix ---
carries the paper's main claim, and a referee is entitled to ask whether it was
shaped to favour the proposed monitor. This module answers that with a
third-party protocol from the incumbent method's own paper.

The scheme: rank features by information gain against the class label, then
after the change point cyclically rotate the values of a chosen subset of
feature columns. Rotating the **top 25%** of features by information gain
degrades the classifier and is a true drift that a detector must catch;
rotating the **bottom 25%** leaves classification essentially untouched and is
an irrelevant change that a detector must ignore. The two experiments are
therefore a detectability test and a false-alarm test, and a method is only
credible if it passes both. Sethi & Kantardzic report HDDDM failing the second
one --- signalling on bottom-25% rotations that cost no accuracy.

This operates in feature space, so it applies to the handcrafted-feature
inspection model, which is the setting their protocol was written for. It does
not transfer to the CNN, whose input is an image; that limitation is stated
rather than worked around, because inventing an "equivalent" image-space
rotation would forfeit the whole point of using someone else's protocol.

No feature re-extraction is needed --- the rotation is a column permutation on
the cached feature matrix --- so these experiments are cheap.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif


def rank_features(
    features: np.ndarray,
    labels,
    seed: int = 0,
) -> np.ndarray:
    """Feature indices ordered by information gain, most informative first."""
    classes = sorted(set(map(str, labels)))
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[str(l)] for l in labels])
    gain = mutual_info_classif(features, y, random_state=seed)
    return np.argsort(-gain)


def select_columns(order: np.ndarray, which: str, fraction: float = 0.25) -> np.ndarray:
    """Top or bottom `fraction` of the ranked features."""
    k = max(2, int(round(fraction * len(order))))
    if which == "top":
        return order[:k]
    if which == "bottom":
        return order[-k:]
    raise ValueError("which must be 'top' or 'bottom'")


def rotate(features: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Cyclically rotate the values held in `columns`, row by row.

    With columns (1, 5, 7) the values become (7, 1, 5) --- the marginal
    distribution of the affected feature set is preserved exactly while the
    association between each feature and the label is destroyed. That is what
    makes it a drift the classifier feels rather than a distributional shift a
    feature-space detector could pick up for free.
    """
    out = np.array(features, dtype=float, copy=True)
    block = out[:, columns]
    out[:, columns] = np.roll(block, shift=1, axis=1)
    return out


def induce(
    features: np.ndarray,
    post_mask: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    """Apply the rotation to rows at or after the change point."""
    out = np.array(features, dtype=float, copy=True)
    rows = np.flatnonzero(np.asarray(post_mask, dtype=bool))
    out[rows] = rotate(out[rows], columns)
    return out
