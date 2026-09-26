"""A Laney-style dispersion correction for compositional monitoring.

Laney (2002) observed that attribute charts assume the only variation is
sampling variation, converted each proportion to a z-score using the
within-subgroup standard deviation, and then *measured* the dispersion of
those z-scores by moving ranges rather than assuming it equals one. His
sigma_Z is, in his words, the relative amount of process variation not
explained by the binomial assumption alone; as subgroup size grows the
sampling component shrinks and the batch-to-batch component dominates.

That is the finding of this study, stated for a single proportion in 2002.
This module implements the multivariate compositional analogue, which does not
appear to exist: Li, Tsung & Zou (2014) give the closest multivariate
categorical chart and assume multinomial sampling throughout, with no
dispersion term.

The construction mirrors Laney's three steps.

1. **Sampling-only covariance, per unit.** For a composition estimated from
   ``n`` items, the multinomial covariance of the proportions is
   ``(diag(p) - p p') / n``. Pushing it through the ILR map with Jacobian
   ``V' diag(p)^-1`` and using ``V'1 = 0`` collapses it to

       Sigma_samp = V' diag(1/p) V / n

   the exact analogue of Laney's ``sigma_p = sqrt(p(1-p)/n)``, and like his it
   varies from unit to unit with both composition and size.

2. **Whiten.** ``w = Sigma_samp^{-1/2} (z - mu)`` is the multivariate z-score.
   Under pure multinomial sampling ``Cov(w) = I``; Laney's step is to doubt
   that and measure it.

3. **Measure the dispersion matrix by successive differences.**

       Sigma_W = sum (w_t - w_{t-1})(w_t - w_{t-1})' / (2 (k-1))

   Moving ranges rather than the raw covariance, for Laney's reason: the
   successive-difference estimator is robust to drift in the mean during
   Phase I, which a pooled covariance would absorb into the dispersion and
   inflate the limits. ``Sigma_W = I`` recovers the uncorrected chart exactly,
   so the correction is a strict generalisation, as Laney's p'-chart reduces
   to the p-chart when sigma_Z = 1.

The charting statistic is then Mahalanobis in the corrected metric, with
``Sigma_samp^{1/2} Sigma_W Sigma_samp^{1/2}`` in place of a covariance
estimated from Phase I scatter alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .composition import ilr_basis


def sampling_covariance(composition: np.ndarray, n: float,
                        basis: np.ndarray | None = None) -> np.ndarray:
    """Multinomial sampling covariance of the ILR coordinates of one unit."""
    p = np.asarray(composition, dtype=float)
    if basis is None:
        basis = ilr_basis(len(p))
    p = np.clip(p, 1e-12, None)
    return (basis @ np.diag(1.0 / p) @ basis.T) / max(n, 1.0)


def _inv_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.atleast_2d(matrix))
    values = np.clip(values, 1e-12, None)
    return vectors @ np.diag(values ** -0.5) @ vectors.T


def _sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.atleast_2d(matrix))
    values = np.clip(values, 0.0, None)
    return vectors @ np.diag(np.sqrt(values)) @ vectors.T


def whiten(coords: np.ndarray, props: np.ndarray, sizes: np.ndarray,
           mean: np.ndarray, basis: np.ndarray | None = None,
           reference: np.ndarray | None = None) -> np.ndarray:
    """Multivariate z-scores: centre, then divide out the unit's sampling
    covariance.

    The reference composition is the Phase I MEAN, not each unit's own
    estimate --- Laney forms ``sigma_p = sqrt(pbar(1-pbar)/n_i)`` from the
    overall mean for exactly this reason. Using each unit's own composition
    puts the same noisy quantity in the numerator and the denominator, which
    biases the measured dispersion downward; it read 0.92 on a pure
    multinomial stream where the correct value is 1.00. Only the unit's SIZE
    is allowed to vary the scaling, which is what makes limits respond to
    subgroup size as Laney requires.
    """
    coords = np.asarray(coords, dtype=float)
    if reference is None:
        reference = np.asarray(props, dtype=float).mean(axis=0)
    out = np.empty_like(coords)
    for t in range(len(coords)):
        cov = sampling_covariance(reference, sizes[t], basis)
        out[t] = _inv_sqrt(cov) @ (coords[t] - mean)
    return out


def dispersion_matrix(w: np.ndarray, shrink: float = 0.0,
                      lag: int = 1) -> np.ndarray:
    """Laney's sigma_Z, as a matrix, from successive differences.

    Returns the identity when the multinomial assumption holds exactly.
    ``shrink`` blends toward the identity, which is worth using when Phase I
    is short: the estimator needs (D-1)(D)/2 entries and a noisy estimate of
    the correction is worse than a mild one.

    ``lag`` is the separation between the differenced units and defaults to
    Laney's adjacent pair. Adjacent differencing assumes consecutive units are
    independent draws. When units are clustered --- successive units drawn
    from the same production lot --- neighbours are more alike than units in
    general, the differences are too small, and the measured dispersion is too
    low. On WM-811K the estimate rises from 1.76 at lag 1 to 2.2-2.3 by lag 4
    and then plateaus, exactly the signature of clustering; on the same stream
    shuffled at the wafer level it sits at 0.97 for every lag, and on the two
    unclustered datasets it is flat in lag. Set ``lag`` beyond the cluster
    size when units carry batch structure. Laney notes the univariate version
    of this: sigma_Z below one indicates positive autocorrelation.
    """
    w = np.asarray(w, dtype=float)
    if len(w) < 2 * lag + 1:
        return np.eye(w.shape[1])
    diffs = w[lag:] - w[:-lag]
    sigma = (diffs.T @ diffs) / (2.0 * len(diffs))
    if shrink > 0:
        sigma = (1 - shrink) * sigma + shrink * np.eye(len(sigma))
    return sigma


@dataclass
class LaneyCoDaFit:
    mean: np.ndarray
    dispersion: np.ndarray
    limit: float
    inflation: float
    basis: np.ndarray
    reference: np.ndarray

    @property
    def dim(self) -> int:
        return len(self.mean)


def calibrate(coords: np.ndarray, props: np.ndarray, sizes: np.ndarray,
              target_far: float = 0.005, shrink: float = 0.0, lag: int = 1,
              n_bootstrap: int = 4000, seed: int = 0) -> LaneyCoDaFit:
    """Fit the corrected chart on Phase I units."""
    coords = np.asarray(coords, dtype=float)
    basis = ilr_basis(props.shape[1])
    mean = coords.mean(axis=0)
    reference = np.asarray(props, dtype=float).mean(axis=0)
    w = whiten(coords, props, sizes, mean, basis, reference)
    sigma_w = dispersion_matrix(w, shrink=shrink, lag=lag)

    # Scalar summary, directly comparable to Laney's sigma_Z: 1.0 means the
    # multinomial assumption is adequate, larger means over-dispersion.
    inflation = float(np.sqrt(np.trace(sigma_w) / len(sigma_w)))

    rng = np.random.default_rng(seed)
    root = _sqrt(sigma_w)
    draws = rng.standard_normal((n_bootstrap, len(mean))) @ root.T
    inv = np.linalg.pinv(sigma_w)
    null = np.einsum("ij,jk,ik->i", draws, inv, draws)
    limit = float(np.quantile(null, 1.0 - target_far))
    return LaneyCoDaFit(mean=mean, dispersion=sigma_w, limit=limit,
                        inflation=inflation, basis=basis, reference=reference)


def run(fit: LaneyCoDaFit, coords: np.ndarray, props: np.ndarray,
        sizes: np.ndarray) -> dict[str, np.ndarray]:
    """Apply the corrected chart to Phase II units."""
    w = whiten(coords, props, sizes, fit.mean, fit.basis, fit.reference)
    inv = np.linalg.pinv(fit.dispersion)
    statistic = np.einsum("ij,jk,ik->i", w, inv, w)
    return {"statistic": statistic, "alarm": statistic > fit.limit,
            "limit": np.full(len(statistic), fit.limit)}


def run_uncorrected(fit: LaneyCoDaFit, coords: np.ndarray, props: np.ndarray,
                    sizes: np.ndarray, target_far: float = 0.005) -> dict:
    """The same chart with the dispersion term forced to the identity.

    This is the comparison that isolates the correction: identical whitening,
    identical everything, with only Laney's measured dispersion replaced by
    the multinomial assumption.
    """
    from scipy import stats

    w = whiten(coords, props, sizes, fit.mean, fit.basis, fit.reference)
    statistic = np.einsum("ij,ij->i", w, w)
    limit = float(stats.chi2.ppf(1.0 - target_far, fit.dim))
    return {"statistic": statistic, "alarm": statistic > limit,
            "limit": np.full(len(statistic), limit)}
