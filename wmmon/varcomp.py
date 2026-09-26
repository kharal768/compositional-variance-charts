"""A variance-components chart for compositional classifier output.

A distribution-free dispersion matrix was compared against a fitted
Dirichlet-multinomial and found the fitted route worse despite estimating its
parameter four times more accurately. The explanation is structural, and it is
worth stating as a property of the family rather than as an empirical finding.

For a Dirichlet-multinomial with n trials and concentration alpha_0, the
covariance of the proportions is

    Cov(p_hat) = (n + alpha_0) / (n (1 + alpha_0)) * (diag(p) - p p'),

a *scalar* multiple of the multinomial covariance. Pushing that through the ILR
map multiplies the sampling covariance by the same scalar. So a
Dirichlet-multinomial can inflate dispersion but cannot change its shape: the
over-dispersion it represents is isotropic relative to sampling variation. Real
streams are not obliged to be.

The natural generative model that can is a two-level one. Write the observed
log-ratio vector for unit t as

    z_t = mu + b_t + e_t,     b_t ~ (0, Sigma_b),    e_t ~ (0, Sigma_samp_t),

where ``b_t`` is the between-unit or batch effect and ``e_t`` is multinomial
sampling noise. ``Sigma_samp_t = V diag(p_bar)^-1 V' / n_t`` is known, and
shrinks with unit size; ``Sigma_b`` does not. This is the compositional
analogue of the variance-components view of attribute charting, and it is what
Laney's sigma_Z approximates with a single number.

Estimation is by moments and needs no optimiser. The successive-difference
covariance of the raw coordinates estimates the total,

    S = sum (z_t - z_{t-l})(z_t - z_{t-l})' / (2 (k - l)),

which under the model equals ``Sigma_b + mean_t Sigma_samp_t``. Subtracting the
known sampling part and projecting to the positive semi-definite cone gives
``Sigma_b``. The lag ``l`` carries over from the dispersion estimator and for
the same reason: adjacent differencing understates the between-unit term when
consecutive units share a batch.

The chart then uses a unit-specific covariance ``Sigma_samp_t + Sigma_b``.
Three properties follow, and each is checked in the tests:

* ``Sigma_b = 0`` recovers the uncorrected multinomial chart exactly;
* limits widen for small units and tighten for large ones, because only the
  sampling term carries ``1/n_t``, and they do not collapse to zero as
  ``n_t`` grows --- which is the failure Laney describes at large subgroups;
* the model can represent dispersion that is anisotropic across balances,
  which the Dirichlet-multinomial family cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .composition import ilr_basis
from .laney_coda import sampling_covariance


def _project_psd(matrix: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Nearest positive semi-definite matrix by eigenvalue clipping.

    The moment estimator subtracts one covariance from another and can return
    small negative eigenvalues when the between-unit term is near zero. Clipping
    is the standard remedy and biases Sigma_b upward slightly, which is the
    conservative direction for a control limit.
    """
    values, vectors = np.linalg.eigh(np.atleast_2d(matrix))
    return vectors @ np.diag(np.clip(values, floor, None)) @ vectors.T


@dataclass
class VarCompFit:
    mean: np.ndarray
    between: np.ndarray
    reference: np.ndarray
    basis: np.ndarray
    limit: float
    lag: int

    @property
    def dim(self) -> int:
        return len(self.mean)

    @property
    def between_fraction(self) -> float:
        """Share of total variance at a reference unit attributable to batch.

        Reported alongside a chart as a one-number diagnostic, comparable to
        Laney's sigma_Z but interpretable as a variance decomposition.
        """
        return float(np.trace(self.between) /
                     max(np.trace(self.between) + np.trace(self._mean_sampling), 1e-12))

    _mean_sampling: np.ndarray = None


def fit_between_em(coords: np.ndarray, mean: np.ndarray,
                   sampling: np.ndarray, iterations: int = 200,
                   tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray, int]:
    """Maximum-likelihood between-unit covariance by EM.

    The moment estimator subtracts the mean sampling covariance from the
    observed total. That is sound only when the sampling part is accurate; on a
    stream of small units with zero-replaced rare parts it is not, and the
    difference can be dominated by negative eigenvalues that the projection to
    the PSD cone then manufactures into a spurious batch term.

    Woodall & Thomas estimate components of variance jointly rather than by
    subtraction. This is the compositional version: treat the model
    ``z_t = mu + b_t + e_t`` with ``b_t ~ N(0, Sigma_b)`` and
    ``e_t ~ N(0, Sigma_samp_t)`` known and unit-specific, and maximise the
    likelihood in ``Sigma_b`` by EM. Each iteration takes the posterior of the
    unobserved ``b_t`` and averages its second moment, so the estimate is a sum
    of positive semi-definite terms and is PSD by construction --- no clipping,
    and no negative eigenvalues to project away.

    Returns the estimate, the re-estimated mean, and the iterations used.
    """
    coords = np.asarray(coords, dtype=float)
    n, p = coords.shape
    sigma_b = np.cov(coords, rowvar=False) * 0.5 + 1e-6 * np.eye(p)
    inv_samp = np.array([np.linalg.pinv(s) for s in sampling])

    for iteration in range(1, iterations + 1):
        inv_b = np.linalg.pinv(sigma_b)
        # Re-estimate the mean with precision weights, since a unit observed
        # with a large sampling covariance should count for less.
        weight = np.array([np.linalg.pinv(sigma_b + sampling[t]) for t in range(n)])
        mean = np.linalg.pinv(weight.sum(axis=0)) @ np.einsum(
            "tij,tj->i", weight, coords)

        posterior = np.array([np.linalg.pinv(inv_b + inv_samp[t]) for t in range(n)])
        m = np.einsum("tij,tjk,tk->ti", posterior, inv_samp, coords - mean)
        updated = (np.einsum("ti,tj->ij", m, m) + posterior.sum(axis=0)) / n
        updated = 0.5 * (updated + updated.T)

        shift = float(np.abs(updated - sigma_b).max())
        sigma_b = updated
        if shift < tol:
            break
    return sigma_b, mean, iteration


def calibrate(coords: np.ndarray, props: np.ndarray, sizes: np.ndarray,
              target_far: float = 0.005, lag: int = 1, n_bootstrap: int = 20000,
              estimator: str = "moment", seed: int = 0) -> VarCompFit:
    coords = np.asarray(coords, dtype=float)
    props = np.asarray(props, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    basis = ilr_basis(props.shape[1])
    mean = coords.mean(axis=0)
    reference = props.mean(axis=0)

    sampling = np.array([sampling_covariance(reference, n, basis) for n in sizes])
    mean_sampling = sampling.mean(axis=0)

    if estimator == "em":
        between, mean, _ = fit_between_em(coords, mean, sampling)
    elif estimator == "moment":
        if len(coords) > 2 * lag:
            diffs = coords[lag:] - coords[:-lag]
            total = (diffs.T @ diffs) / (2.0 * len(diffs))
        else:
            total = np.cov(coords, rowvar=False)
        between = _project_psd(np.atleast_2d(total) - mean_sampling)
    else:
        raise ValueError("estimator must be 'moment' or 'em'")

    # The statistic is Mahalanobis in a unit-specific covariance, so its null
    # is not exactly chi-square when unit sizes vary. The limit is taken from a
    # simulation over the observed size distribution.
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(max(1, n_bootstrap // len(sizes))):
        for n in sizes:
            cov = sampling_covariance(reference, n, basis) + between
            root = np.linalg.cholesky(cov + 1e-12 * np.eye(len(cov)))
            x = root @ rng.standard_normal(len(cov))
            draws.append(float(x @ np.linalg.pinv(cov) @ x))
    limit = float(np.quantile(np.asarray(draws), 1.0 - target_far))

    fit = VarCompFit(mean=mean, between=between, reference=reference, basis=basis,
                     limit=limit, lag=lag)
    fit._mean_sampling = mean_sampling
    return fit


def run(fit: VarCompFit, coords: np.ndarray, props: np.ndarray,
        sizes: np.ndarray) -> dict[str, np.ndarray]:
    coords = np.asarray(coords, dtype=float)
    statistic = np.empty(len(coords))
    cache: dict[float, np.ndarray] = {}
    for t, n in enumerate(np.asarray(sizes, dtype=float)):
        if n not in cache:
            cov = sampling_covariance(fit.reference, n, fit.basis) + fit.between
            cache[n] = np.linalg.pinv(cov)
        d = coords[t] - fit.mean
        statistic[t] = float(d @ cache[n] @ d)
    return {"statistic": statistic, "alarm": statistic > fit.limit,
            "limit": np.full(len(statistic), fit.limit)}
