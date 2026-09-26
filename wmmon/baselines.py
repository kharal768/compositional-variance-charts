"""Competing detectors.

These are the comparisons a manufacturing-systems reviewer will expect, and
they are deliberately not all control charts.  If the ILR monitor does not beat
ADWIN and a two-sample test on the classifier's own confidence scores, that is
a finding to report, not a result to bury.

All detectors share one interface: given the per-lot stream they return a
boolean alarm array of the same length, so the same
``monitor.detection_metrics`` is used for every method.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def page_hinkley(
    values: np.ndarray,
    delta: float = 0.005,
    threshold: float = 50.0,
    burn_in: int = 30,
) -> np.ndarray:
    """Page-Hinkley test on a univariate summary of the stream."""
    values = np.asarray(values, dtype=float)
    alarm = np.zeros(len(values), dtype=bool)
    if len(values) <= burn_in:
        return alarm

    baseline = values[:burn_in].mean()
    cumulative = 0.0
    minimum = 0.0
    for t in range(burn_in, len(values)):
        cumulative += values[t] - baseline - delta
        minimum = min(minimum, cumulative)
        if cumulative - minimum > threshold:
            alarm[t] = True
    return alarm


def adwin(
    values: np.ndarray,
    confidence: float = 0.002,
    min_window: int = 10,
    max_window: int = 400,
) -> np.ndarray:
    """ADWIN-style adaptive windowing (exact-window reference implementation).

    Slower than the bucket-based original but exact, which matters when the
    comparison is the point.  Streams here are thousands of lots, not millions
    of records, so the cost is irrelevant.
    """
    values = np.asarray(values, dtype=float)
    alarm = np.zeros(len(values), dtype=bool)
    window: list[float] = []

    for t, value in enumerate(values):
        window.append(float(value))
        if len(window) > max_window:
            window = window[-max_window:]
        if len(window) < 2 * min_window:
            continue

        arr = np.asarray(window)
        n = len(arr)
        variance = arr.var(ddof=1) + 1e-12
        cut_found = False
        for split in range(min_window, n - min_window + 1):
            left, right = arr[:split], arr[split:]
            harmonic = 1.0 / (1.0 / len(left) + 1.0 / len(right))
            delta_prime = confidence / max(np.log(n), 1.0)
            epsilon = np.sqrt(
                2.0 * variance * np.log(2.0 / delta_prime) / harmonic
            ) + 2.0 * np.log(2.0 / delta_prime) / (3.0 * harmonic)
            if abs(left.mean() - right.mean()) > epsilon:
                cut_found = True
                window = list(right)
                break
        if cut_found:
            alarm[t] = True
    return alarm


def ks_two_sample(
    values: np.ndarray,
    reference_size: int = 100,
    window: int = 30,
    alpha: float = 0.005,
) -> np.ndarray:
    """Kolmogorov-Smirnov test of a sliding window against a fixed reference."""
    values = np.asarray(values, dtype=float)
    alarm = np.zeros(len(values), dtype=bool)
    if len(values) <= reference_size + window:
        return alarm
    reference = values[:reference_size]
    for t in range(reference_size + window, len(values)):
        current = values[t - window : t]
        if stats.ks_2samp(reference, current).pvalue < alpha:
            alarm[t] = True
    return alarm


def _rbf_mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    def kernel(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d2)

    kxx, kyy, kxy = kernel(x, x), kernel(y, y), kernel(x, y)
    m, n = len(x), len(y)
    return (
        (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
        + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
        - 2.0 * kxy.mean()
    )


def mmd_detector(
    features: np.ndarray,
    reference_size: int = 100,
    window: int = 30,
    n_permutations: int = 120,
    alpha: float = 0.005,
    stride: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Multivariate MMD two-sample test with a permutation null."""
    features = np.asarray(features, dtype=float)
    alarm = np.zeros(len(features), dtype=bool)
    if len(features) <= reference_size + window:
        return alarm

    reference = features[:reference_size]
    pooled = np.vstack([reference, features[reference_size : reference_size + window]])
    distances = ((pooled[:, None, :] - pooled[None, :, :]) ** 2).sum(-1)
    median = np.median(distances[distances > 0])
    gamma = 1.0 / max(median, 1e-9)

    rng = np.random.default_rng(seed)
    # Permutation MMD is the most expensive baseline; evaluating every
    # ``stride`` lots keeps it tractable. The stride is reported alongside the
    # detection delay so the comparison stays honest: a stride of k inflates
    # this detector's delay by up to k-1 lots, which must be stated.
    for t in range(reference_size + window, len(features), stride):
        current = features[t - window : t]
        observed = _rbf_mmd2(reference, current, gamma)
        combined = np.vstack([reference, current])
        null = np.empty(n_permutations)
        for b in range(n_permutations):
            perm = rng.permutation(len(combined))
            null[b] = _rbf_mmd2(
                combined[perm[: len(reference)]], combined[perm[len(reference) :]], gamma
            )
        if (null >= observed).mean() < alpha:
            alarm[t] = True
    return alarm


def proportion_mewma(
    proportions: np.ndarray,
    sizes: np.ndarray,
    lam: float = 0.2,
    phase_one: int = 200,
    target_far: float = 0.005,
    seed: int = 0,
) -> np.ndarray:
    """MEWMA applied directly to raw proportions, with no log-ratio transform.

    This is the comparison that carries the methodological argument: raw
    proportions live on the simplex, so their covariance is singular by
    construction (rows sum to one) and Euclidean distances between them are not
    meaningful.  The pseudo-inverse below keeps it numerically alive; whether
    that costs detection power is an empirical question the experiment answers.
    """
    from .monitor import _mewma_statistics

    proportions = np.asarray(proportions, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    reference = proportions[:phase_one]
    mean = reference.mean(axis=0)

    scale = np.sqrt(sizes[:phase_one] / np.median(sizes))
    standardised = (reference - mean) * scale[:, None]
    cov = np.cov(standardised, rowvar=False)
    cov_inv = np.linalg.pinv(cov)

    rng = np.random.default_rng(seed)
    p = proportions.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    root = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    null = np.concatenate(
        [
            _mewma_statistics(rng.standard_normal((400, p)) @ root.T, cov_inv, lam)
            for _ in range(50)
        ]
    )
    limit = float(np.quantile(null[50:], 1.0 - target_far))

    full_scale = np.sqrt(sizes / np.median(sizes))
    full = (proportions - mean) * full_scale[:, None]
    return _mewma_statistics(full, cov_inv, lam) > limit


def max_confidence_stream(probabilities: np.ndarray, lots, lot_sequence) -> np.ndarray:
    """Mean maximum-softmax confidence per lot.

    The standard cheap signal for model degradation, and the honest baseline to
    beat: if mean confidence detects the same shifts as a compositional monitor,
    a fab has no reason to deploy the more complex method.
    """
    lots = np.asarray(lots, dtype=object)
    confidence = probabilities.max(axis=1)
    index: dict[str, list[int]] = {}
    for position, lot in enumerate(lots):
        index.setdefault(str(lot), []).append(position)
    return np.asarray(
        [
            confidence[index[str(lot)]].mean()
            for lot in lot_sequence
            if str(lot) in index
        ]
    )
