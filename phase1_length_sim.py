#!/usr/bin/env python3
"""How the plain Phase I covariance chart's in-control rate depends on the length of Phase I.

An individuals T^2 with the Phase I sample covariance and an empirical limit estimates eight
means, thirty-six covariance entries and an extreme quantile from the Phase I units, so its
realised in-control rate depends on how many there are (Li, Tsung & Zou, 2014). The real-stream
sensitivity to the Phase I fraction cannot be tabulated without the data; this reports the
effect in simulation, with known truth, at a fixed unit size and the between-unit covariance used
throughout. Streams are independent, so the rate is marginal over Phase I draws.
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
NOMINAL, STREAMS, EVAL = 5.0, 80, 400
LENGTHS = (120, 200, 240, 400, 800)


def main() -> int:
    p = np.array([0.808, 0.053, 0.028, 0.024, 0.020, 0.017, 0.017, 0.017, 0.017])
    p = p / p.sum()
    V = ilr_basis(len(p)); dim = len(p) - 1
    base = sampling_covariance(p, 25.0, V)
    B = np.diag(np.linspace(1.0, 0.3, dim)); B *= 0.30 * np.trace(base) / np.trace(B)
    rows = []
    print(f"{STREAMS} streams per length, {EVAL} Phase II units each, nominal {NOMINAL:.1f} per 1,000\n")
    print(f"{'Phase I units':>14}{'empirical':>11}{'F limit':>10}{'ratio (F)':>11}{'95% CI (F)':>18}")
    for L in LENGTHS:
        alarms, per, alarms_f = 0, [], 0
        for s in range(STREAMS):
            rng = np.random.default_rng(5000 + 17 * L + s)
            b = rng.multivariate_normal(np.zeros(dim), B, size=L + EVAL)
            counts = rng.multinomial(25, p, size=L + EVAL).astype(float)
            z = ilr(bayesian_multiplicative_replacement(counts)) + b
            ref = z[:L]
            mu = ref.mean(axis=0)
            inv = np.linalg.pinv(np.cov(ref, rowvar=False))
            stat = np.einsum("ij,jk,ik->i", z - mu, inv, z - mu)
            limit = float(np.quantile(stat[:L], 1 - NOMINAL / 1000))
            k = int((stat[L:] > limit).sum())
            alarms += k; per.append(k)
            # the textbook Phase II limit for an individuals T^2 with estimated mean and
            # covariance, which assumes multivariate normality and uses an F quantile
            d_ = dim
            f_limit = (d_ * (L + 1) * (L - 1) / (L * (L - d_))) * stats.f.ppf(
                1 - NOMINAL / 1000, d_, L - d_)
            alarms_f += int((stat[L:] > f_limit).sum())
        n = STREAMS * EVAL
        lo = stats.beta.ppf(0.025, alarms, n - alarms + 1) * 1000 if alarms else 0.0
        hi = stats.beta.ppf(0.975, alarms + 1, n - alarms) * 1000
        rate = 1000.0 * alarms / n
        lo_f = stats.beta.ppf(0.025, alarms_f, n - alarms_f + 1) * 1000 if alarms_f else 0.0
        hi_f = stats.beta.ppf(0.975, alarms_f + 1, n - alarms_f) * 1000
        rate_f = 1000.0 * alarms_f / n
        rows.append({"phase_one": L, "alarms": alarms, "exposure": n, "rate": rate,
                     "f_rate": rate_f, "f_ratio": rate_f / NOMINAL, "f_ci": [lo_f, hi_f],
                     "ratio": rate / NOMINAL, "ci": [lo, hi],
                     "share_streams_silent": float(np.mean(np.array(per) == 0))})
        print(f"{L:>14}{rate:>11.2f}{rate_f:>10.2f}{rate_f/NOMINAL:>11.2f}{f'[{lo_f:.2f}, {hi_f:.2f}]':>18}")
    RESULTS.mkdir(exist_ok=True)
    json.dump({"nominal_per_1000": NOMINAL, "streams": STREAMS, "rows": rows},
              open(RESULTS / "phase1_length_sim.json", "w"), indent=2)
    print("\nwrote results/phase1_length_sim.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
