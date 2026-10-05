#!/usr/bin/env python3
"""How the plain chart's F limit depends on the amount of Gaussian between-unit variation.

The textbook limit assumes multivariate normality. The log-ratio coordinates of multinomial counts
with rare classes are skewed and heavy-tailed, so on a stream with no between-unit variation the
statistic's upper tail exceeds the chi-square or F quantile. The simulated streams elsewhere add a
Gaussian between-unit component, which pulls the total towards normality. This varies its size, as
a multiple of the baseline used throughout (0 is a pure multinomial stream), at a fixed unit size.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon.composition import bayesian_multiplicative_replacement, ilr, ilr_basis
from wmmon.laney_coda import sampling_covariance
from wmmon.paths import WMMON_HOME

RESULTS = Path(f"{WMMON_HOME}/results")
NOMINAL, STREAMS, T, PHASE_ONE, WARMUP, UNIT = 5.0, 300, 600, 240, 20, 24
P = np.array([0.808, 0.053, 0.028, 0.024, 0.020, 0.017, 0.017, 0.017, 0.017]); P = P / P.sum()
V = ilr_basis(len(P)); DIM = len(P) - 1
SCALES = (0.0, 0.25, 1.0, 4.0)


def main() -> int:
    base = sampling_covariance(P, 25.0, V)
    B0 = np.diag(np.linspace(1.0, 0.3, DIM)); B0 *= 0.30 * np.trace(base) / np.trace(B0)
    f_lim = (DIM * (PHASE_ONE + 1) * (PHASE_ONE - 1) / (PHASE_ONE * (PHASE_ONE - DIM))) * stats.f.ppf(
        1 - NOMINAL / 1000, DIM, PHASE_ONE - DIM)
    rows = []
    print(f"{'between-unit scale':>20}{'share of variance':>19}{'rate':>8}{'95% CI':>16}")
    for sc in SCALES:
        B = sc * B0
        alarms = exposure = 0
        for s in range(STREAMS):
            rng = np.random.default_rng(30000 + s)
            b = rng.multivariate_normal(np.zeros(DIM), B, size=T) if sc > 0 else 0.0
            counts = rng.multinomial(UNIT, P, size=T).astype(float)
            z = ilr(bayesian_multiplicative_replacement(counts)) + b
            mu = z[:PHASE_ONE].mean(axis=0)
            inv = np.linalg.pinv(np.cov(z[:PHASE_ONE], rowvar=False))
            d = z - mu
            stat = np.einsum("ij,jk,ik->i", d, inv, d)[PHASE_ONE:][WARMUP:]
            alarms += int((stat > f_lim).sum()); exposure += len(stat)
        samp = np.trace(sampling_covariance(P, float(UNIT), V))
        share = np.trace(B) / (np.trace(B) + samp)
        lo = stats.beta.ppf(0.025, alarms, exposure - alarms + 1) * 1000 if alarms else 0.0
        hi = stats.beta.ppf(0.975, alarms + 1, exposure - alarms) * 1000
        rows.append({"between_scale": sc, "between_share_closed_form": float(share), "alarms": alarms,
                     "exposure": exposure, "rate": 1000.0 * alarms / exposure, "ci": [lo, hi]})
        print(f"{sc:>20.2f}{100*share:>18.0f}%{1000.0*alarms/exposure:>8.2f}{f'[{lo:.2f}, {hi:.2f}]':>16}")
    RESULTS.mkdir(exist_ok=True)
    json.dump({"unit": UNIT, "streams": STREAMS, "phase_one": PHASE_ONE, "rows": rows},
              open(RESULTS / "f_limit_between_variance.json", "w"), indent=2)
    print("\nwrote results/f_limit_between_variance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
