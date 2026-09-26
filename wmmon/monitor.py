"""The monitoring layer: a rate-invariant MEWMA on ILR coordinates.

Design note, because this is the part a reviewer will press on.  A composition
is scale-invariant by construction, so the *mean* of the ILR coordinates does
not depend on how many wafers a lot contains.  Its *sampling variance* does:
it scales roughly as 1/n.  A monitor that ignores this inflates false alarms on
short lots and loses power on full ones, which is exactly the failure mode that
shows up when production rate or lot size varies.  The fix implemented here is
to standardise each observation by its own lot size before the EWMA recursion,
so the in-control distribution of the charting statistic is the same whatever
the lot size.  That is the rate-invariance property, and the ablation in
:func:`run_monitor` (``rate_invariant=False``) is what demonstrates it matters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

log = logging.getLogger(__name__)


@dataclass
class MonitorFit:
    """In-control model estimated from Phase I."""

    mean: np.ndarray
    cov: np.ndarray
    cov_inv: np.ndarray
    lam: float
    limit: float
    rate_invariant: bool
    reference_lot_size: float
    target_far: float
    achieved_far: float
    block_length: int
    lag1_dependence: float = 0.0
    phi: np.ndarray | None = None

    @property
    def dim(self) -> int:
        return len(self.mean)


def _shrunk_covariance(residuals: np.ndarray, shrinkage: float | None) -> np.ndarray:
    """Sample covariance with optional Ledoit-Wolf-style shrinkage to a
    scaled identity.  With nine classes the ILR space is 8-dimensional, so the
    sample covariance is usually fine; shrinkage guards small Phase I windows."""
    cov = np.cov(residuals, rowvar=False)
    cov = np.atleast_2d(cov)
    if shrinkage is None:
        try:
            from sklearn.covariance import LedoitWolf

            return LedoitWolf().fit(residuals).covariance_
        except Exception:
            shrinkage = 0.0
    target = np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    return (1.0 - shrinkage) * cov + shrinkage * target


def _fit_var1(residuals: np.ndarray) -> np.ndarray:
    """Least-squares VAR(1) coefficient matrix for Phase I residuals."""
    past, future = residuals[:-1], residuals[1:]
    phi, *_ = np.linalg.lstsq(past, future, rcond=None)
    return phi.T


def _lag1_dependence(series: np.ndarray) -> float:
    """Scalar summary of lag-1 dependence: mean |correlation| across coordinates."""
    if len(series) < 3:
        return 0.0
    past, future = series[:-1], series[1:]
    values = []
    for j in range(series.shape[1]):
        a, b = past[:, j], future[:, j]
        sa, sb = a.std(), b.std()
        if sa > 1e-12 and sb > 1e-12:
            values.append(abs(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(values)) if values else 0.0


def _decide_prewhitening(
    standardised: np.ndarray,
    mode,
    min_reduction: float = 0.25,
) -> np.ndarray | None:
    """Fit a VAR(1) filter only when it is actually warranted.

    Prewhitening is not free.  On a stream with no lot-to-lot dependence the
    fitted filter is pure estimation noise: it inflates the innovation variance
    while the Phase I covariance is estimated in-sample on the same residuals,
    so the control limit comes out too low and the false-alarm rate explodes.
    That failure is easy to miss because the Phase I diagnostics still look
    healthy.  So the filter is fitted on part of Phase I, judged on the rest,
    and kept only if it measurably reduces lag-1 dependence out of sample.
    """
    if mode is False:
        return None
    n, p = standardised.shape
    if n < max(60, 20 * p):
        # Too few lots to estimate p^2 coefficients; forcing it would overfit.
        return None

    cut = int(0.7 * n)
    phi_trial = _fit_var1(standardised[:cut])
    holdout = standardised[cut:]
    before = _lag1_dependence(holdout)
    after = _lag1_dependence(_prewhiten(holdout, phi_trial))

    if mode == "auto" and not (before > 0.05 and after < (1 - min_reduction) * before):
        return None
    return _fit_var1(standardised)


def _prewhiten(series: np.ndarray, phi: np.ndarray | None) -> np.ndarray:
    """Innovations of a VAR(1) filter; identity when ``phi`` is None.

    Lot-to-lot dependence is real in a fab: a drifting tool touches
    consecutive lots, so ILR coordinates are autocorrelated. Charting the raw
    coordinates inflates false alarms no matter how the limit is calibrated,
    because the EWMA compounds the dependence. Monitoring the innovations of a
    fitted VAR(1) is the standard remedy and is what makes the realised
    false-alarm rate match the nominal one.
    """
    if phi is None:
        return series
    out = series.copy()
    out[1:] = series[1:] - series[:-1] @ phi.T
    return out


def _standardise(
    coords: np.ndarray,
    sizes: np.ndarray,
    mean: np.ndarray,
    rate_invariant: bool,
    reference_lot_size: float,
) -> np.ndarray:
    """Centre, and rescale by sqrt(n / n_ref) when rate-invariance is on."""
    centred = coords - mean
    if not rate_invariant:
        return centred
    scale = np.sqrt(np.asarray(sizes, dtype=float) / reference_lot_size)
    return centred * scale[:, None]


def _mewma_statistics(
    standardised: np.ndarray,
    cov_inv: np.ndarray,
    lam: float,
) -> np.ndarray:
    """MEWMA T^2 sequence with the exact time-varying covariance factor."""
    n, p = standardised.shape
    z = np.zeros(p)
    out = np.empty(n)
    for t in range(n):
        z = lam * standardised[t] + (1.0 - lam) * z
        factor = (lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** (2 * (t + 1)))
        out[t] = float(z @ cov_inv @ z) / factor
    return out


def calibrate(
    coords: np.ndarray,
    sizes: np.ndarray,
    lam: float = 0.2,
    target_far: float = 0.005,
    rate_invariant: bool = True,
    prewhiten: bool | str = "auto",
    shrinkage: float | None = None,
    n_bootstrap: int = 2000,
    block_length: int | None = None,
    seed: int = 0,
) -> MonitorFit:
    """Estimate the in-control model and the control limit from Phase I lots.

    The limit is set by parametric bootstrap rather than from an asymptotic
    chi-square quantile: the MEWMA statistic is serially dependent by
    construction, so the marginal chi-square limit does not deliver the nominal
    false-alarm rate, and using it is a standard way to get a chart that looks
    calibrated and is not.

    ``target_far`` is the false-alarm probability per lot.  At 0.005 and a
    typical fab throughput this is roughly five false alarms per 1,000 lots.
    """
    coords = np.asarray(coords, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    if coords.ndim != 2 or len(coords) < 30:
        raise ValueError("Phase I needs at least 30 lots")

    mean = coords.mean(axis=0)
    reference = float(np.median(sizes))
    standardised = _standardise(coords, sizes, mean, rate_invariant, reference)
    dependence = _lag1_dependence(standardised)
    phi = _decide_prewhitening(standardised, prewhiten)
    standardised = _prewhiten(standardised, phi)
    cov = _shrunk_covariance(standardised, shrinkage)
    cov_inv = np.linalg.pinv(cov)

    # Bootstrap null.  A parametric Gaussian draw was the obvious choice here
    # and it is wrong: the ILR stream carries serial dependence (a tool state
    # persists across consecutive lots) and heavier-than-normal tails, so an
    # iid Gaussian null reproduces the asymptotic chi-square limit almost
    # exactly and delivers several times the nominal false-alarm rate in
    # practice.  The moving-block bootstrap below resamples *observed* Phase I
    # residuals in contiguous blocks, which preserves short-range dependence.
    # With ``prewhiten=True`` the dependence has already been removed, so
    # ``block_length=1`` (an iid resample of the empirical innovations) is
    # correct and still handles the heavy tails that the Gaussian null misses.
    # Blocks longer than 1 are for the no-prewhitening ablation.
    rng = np.random.default_rng(seed)
    n_lots, p = standardised.shape
    if block_length is None:
        # No filter but visible dependence -> resample in blocks so the null
        # inherits it. Filter fitted -> innovations are near-white, blocks of 1.
        block_length = 1 if phi is not None or dependence <= 0.05 else 10
    block_length = max(1, min(block_length, n_lots // 4))
    series_length = max(n_lots, 400)

    null_stats = []
    n_series = max(1, n_bootstrap // 20)
    n_blocks = int(np.ceil(series_length / block_length))
    max_start = n_lots - block_length
    for _ in range(n_series):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        draws = np.concatenate(
            [standardised[s : s + block_length] for s in starts]
        )[:series_length]
        null_stats.append(_mewma_statistics(draws, cov_inv, lam))
    null = np.concatenate(null_stats)
    # Discard the EWMA burn-in, where the statistic is not yet stationary.
    burn = min(50, len(null) // 10)
    null = null[burn:]

    limit = float(np.quantile(null, 1.0 - target_far))
    achieved = float((null > limit).mean())

    if phi is None and dependence > 0.05 and n_lots < 20 * p:
        log.warning(
            "Phase I has %d lots with lag-1 dependence %.2f but needs >= %d to "
            "fit the VAR(1) filter; block resampling alone will not hold the "
            "nominal false-alarm rate here. Extend Phase I before reporting.",
            n_lots, dependence, 20 * p,
        )

    return MonitorFit(
        mean=mean,
        cov=cov,
        cov_inv=cov_inv,
        lam=lam,
        limit=limit,
        rate_invariant=rate_invariant,
        reference_lot_size=reference,
        target_far=target_far,
        achieved_far=achieved,
        block_length=block_length,
        lag1_dependence=dependence,
        phi=phi,
    )


def run_monitor(
    fit: MonitorFit,
    coords: np.ndarray,
    sizes: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply a fitted monitor to a Phase II stream."""
    standardised = _standardise(
        np.asarray(coords, dtype=float),
        np.asarray(sizes, dtype=float),
        fit.mean,
        fit.rate_invariant,
        fit.reference_lot_size,
    )
    standardised = _prewhiten(standardised, fit.phi)
    statistic = _mewma_statistics(standardised, fit.cov_inv, fit.lam)
    return {
        "statistic": statistic,
        "limit": np.full(len(statistic), fit.limit),
        "alarm": statistic > fit.limit,
    }


def diagnose(
    fit: MonitorFit,
    coords: np.ndarray,
    sizes: np.ndarray,
    position: int,
) -> np.ndarray:
    """Per-coordinate contribution to the statistic at one alarm position.

    Decomposes the Mahalanobis form so the operator sees *which balance* moved,
    which is what makes the alarm actionable on a line.  Returns contributions
    summing to the statistic value.
    """
    standardised = _standardise(
        np.asarray(coords, dtype=float),
        np.asarray(sizes, dtype=float),
        fit.mean,
        fit.rate_invariant,
        fit.reference_lot_size,
    )
    standardised = _prewhiten(standardised, fit.phi)
    lam = fit.lam
    z = np.zeros(fit.dim)
    for t in range(position + 1):
        z = lam * standardised[t] + (1.0 - lam) * z
    factor = (lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** (2 * (position + 1)))
    weighted = fit.cov_inv @ z
    return (z * weighted) / factor


def detection_metrics(
    alarm: np.ndarray,
    change_point: int | None,
    warmup: int = 0,
) -> dict[str, float]:
    """Detection delay and false-alarm rate, in lots.

    ``change_point`` is the first out-of-control lot index, or ``None`` for a
    stream that is in control throughout.  Delay is counted in lots from the
    change point to the first alarm at or after it; ``inf`` when never detected.
    """
    alarm = np.asarray(alarm, dtype=bool)
    n = len(alarm)
    if change_point is None:
        false_alarms = int(alarm[warmup:].sum())
        exposure = max(n - warmup, 1)
        return {
            "false_alarms": float(false_alarms),
            "far_per_lot": false_alarms / exposure,
            "far_per_1000_lots": 1000.0 * false_alarms / exposure,
            "detection_delay": float("nan"),
            "detected": float("nan"),
        }

    pre = alarm[warmup:change_point]
    exposure = max(change_point - warmup, 1)
    false_alarms = int(pre.sum())

    post = np.flatnonzero(alarm[change_point:])
    delay = float(post[0]) if len(post) else float("inf")
    return {
        "false_alarms": float(false_alarms),
        "far_per_lot": false_alarms / exposure,
        "far_per_1000_lots": 1000.0 * false_alarms / exposure,
        "detection_delay": delay,
        "detected": float(np.isfinite(delay)),
    }


def chi_square_limit(dim: int, target_far: float) -> float:
    """The naive asymptotic limit, kept only so the paper can show the gap
    between it and the bootstrap limit."""
    return float(stats.chi2.ppf(1.0 - target_far, dim))


def _episodes(alarm: np.ndarray, merge_gap: int = 1) -> list[tuple[int, int]]:
    """Contiguous runs of alarms, merged across gaps of ``merge_gap`` or less."""
    positions = np.flatnonzero(np.asarray(alarm, dtype=bool))
    if not len(positions):
        return []
    runs, start, prev = [], positions[0], positions[0]
    for p in positions[1:]:
        if p - prev <= merge_gap:
            prev = p
            continue
        runs.append((int(start), int(prev)))
        start = prev = p
    runs.append((int(start), int(prev)))
    return runs


def episode_metrics(
    alarm: np.ndarray,
    change_point: int | None,
    warmup: int = 0,
    merge_gap: int = 1,
) -> dict[str, float]:
    """Detection metrics counted in episodes rather than individual alarms.

    An EWMA statistic stays above its limit for several lots after a single
    excursion, so counting raw alarms multiplies one event into many and
    overstates the false-alarm rate. Consecutive alarms are therefore merged
    into one episode, which is also how a line operator experiences them:
    one investigation, not ten.
    """
    alarm = np.asarray(alarm, dtype=bool)
    n = len(alarm)

    if change_point is None:
        runs = _episodes(alarm[warmup:], merge_gap)
        exposure = max(n - warmup, 1)
        return {
            "false_alarm_episodes": float(len(runs)),
            "episodes_per_1000_lots": 1000.0 * len(runs) / exposure,
            "detection_delay": float("nan"),
            "detected": float("nan"),
        }

    pre_runs = _episodes(alarm[warmup:change_point], merge_gap)
    exposure = max(change_point - warmup, 1)
    post = np.flatnonzero(alarm[change_point:])
    delay = float(post[0]) if len(post) else float("inf")
    return {
        "false_alarm_episodes": float(len(pre_runs)),
        "episodes_per_1000_lots": 1000.0 * len(pre_runs) / exposure,
        "detection_delay": delay,
        "detected": float(np.isfinite(delay)),
    }
