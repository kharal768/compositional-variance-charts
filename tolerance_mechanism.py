#!/usr/bin/env python3
"""What actually breaks the chart when the known term is wrong?

Section 6.9 scaled the closed-form sampling covariance by k and found failures
at k below one, which was read as "understating the known term is not absorbed".
That reading assumes the closed form is right at k = 1. Section 4.4 shows it is
not: at these unit sizes its trace is several times the empirical one. So k = 0.5
moves the assumed term towards the truth, not away from it, and the earlier
explanation cannot be correct.

This measures misspecification against the truth rather than against the closed
form. A stream is simulated with a known between-unit covariance and a sampling
distribution whose true covariance is obtained by Monte Carlo at the same
composition and unit size. The chart is then run with an assumed sampling term
scaled by k, and the realised in-control rate recorded against the ratio of
assumed to true trace. Two families are compared: scaling the whole term, and
scaling one balance only, which separates size from shape.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon import varcomp as VC
from wmmon.composition import bayesian_multiplicative_replacement, ilr, ilr_basis
from wmmon.laney_coda import sampling_covariance
from wmmon.paths import WMMON_HOME

RESULTS = Path(f"{WMMON_HOME}/results")
NOMINAL, T, N_UNIT = 5.0, 1400, 24
DRAWS = 120_000


def true_sampling(p, n, seed=0):
    """The covariance the coordinates actually have, zero replacement included."""
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, p, size=DRAWS).astype(float)
    return np.cov(ilr(bayesian_multiplicative_replacement(counts)), rowvar=False)


def main() -> int:
    rng = np.random.default_rng(0)
    p = np.array([0.808, 0.053, 0.028, 0.024, 0.020, 0.017, 0.017, 0.017, 0.017])
    p = p / p.sum()
    V = ilr_basis(len(p))
    closed = sampling_covariance(p, float(N_UNIT), V)
    truth = true_sampling(p, N_UNIT)
    dim = closed.shape[0]
    print(f"closed-form trace {np.trace(closed):.3f} | true trace {np.trace(truth):.3f} "
          f"| closed/true {np.trace(closed)/np.trace(truth):.2f}x\n")

    # a stream with genuine between-unit variation and no drift
    B = np.diag(np.linspace(1.0, 0.3, dim))
    B *= 0.30 * np.trace(truth) / np.trace(B)
    b = rng.multivariate_normal(np.zeros(dim), B, size=T)
    counts = np.array([rng.multinomial(N_UNIT, p) for _ in range(T)], dtype=float)
    coords = ilr(bayesian_multiplicative_replacement(counts)) + b
    props = np.tile(p, (T, 1))
    sizes = np.full(T, float(N_UNIT))
    phase_one, warmup = int(0.4 * T), 20
    exposure = T - phase_one - warmup

    rows = []
    print(f"{'family':>10}{'k':>7}{'assumed/true':>14}{'rate':>8}{'95% CI':>17}{'covers':>8}")
    for family in ("whole term", "one balance"):
        for k in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
            M = np.eye(dim)
            if family == "whole term":
                M *= np.sqrt(k)
            else:
                M[0, 0] = np.sqrt(k)
            assumed = M @ closed @ M.T
            ratio = float(np.trace(assumed) / np.trace(truth))
            import wmmon.varcomp as _V
            orig = _V.sampling_covariance
            _V.sampling_covariance = lambda ref, n, basis, _a=assumed: _a
            try:
                fit = _V.calibrate(coords[:phase_one], props[:phase_one], sizes[:phase_one],
                                   target_far=NOMINAL / 1000.0, estimator="em", seed=0)
                out = _V.run(fit, coords[phase_one:], props[phase_one:], sizes[phase_one:])
            finally:
                _V.sampling_covariance = orig
            kk = int(out["alarm"][warmup:].sum())
            lo = stats.beta.ppf(0.025, kk, exposure - kk + 1) * 1000 if kk else 0.0
            hi = stats.beta.ppf(0.975, kk + 1, exposure - kk) * 1000
            covers = lo <= NOMINAL <= hi
            rows.append({"family": family, "k": k, "assumed_over_true": ratio,
                         "rate": 1000.0 * kk / exposure, "ci": [lo, hi],
                         "covers": bool(covers)})
            print(f"{family:>10}{k:>7.2f}{ratio:>14.2f}{1000.0*kk/exposure:>8.1f}"
                  f"{f'[{lo:.1f}, {hi:.1f}]':>17}{'yes' if covers else 'NO':>8}")
    RESULTS.mkdir(exist_ok=True)
    json.dump({"nominal_per_1000": NOMINAL, "unit": N_UNIT, "T": T,
               "closed_over_true_trace": float(np.trace(closed) / np.trace(truth)),
               "rows": rows}, open(RESULTS / "tolerance_mechanism.json", "w"), indent=2)
    print("\nwrote results/tolerance_mechanism.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
