#!/usr/bin/env python3
"""The plain individuals T^2 with the textbook F limit: marginal rate, spread, and heteroscedasticity.

Three questions the simulated comparisons leave open.

1. Spread. The F limit fixes the marginal rate, but a practitioner runs one chart on one stream with
   one estimated mean and covariance. The distribution of the conditional in-control rate across
   independent Phase I draws is what that practitioner experiences.
2. Exposure under heteroscedasticity. With varying unit sizes the units have different covariances, so
   T^2 is not exactly F-distributed. The earlier check rested on 940 units; this pools several hundred
   streams.
3. Size. The rate by unit-size class, because a marginal rate on target can hide small units alarming
   too often and large units too rarely.

No EM fit is needed, so many streams are affordable. Phase I is 240 units, as in the exposure sweep.
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
NOMINAL, STREAMS, T, PHASE_ONE, WARMUP = 5.0, 300, 600, 240, 20
P = np.array([0.808, 0.053, 0.028, 0.024, 0.020, 0.017, 0.017, 0.017, 0.017]); P = P / P.sum()
V = ilr_basis(len(P)); DIM = len(P) - 1


def main() -> int:
    base = sampling_covariance(P, 25.0, V)
    B = np.diag(np.linspace(1.0, 0.3, DIM)); B *= 0.30 * np.trace(base) / np.trace(B)
    designs = {"fixed, 24 items": lambda rng: np.full(T, 24),
               "fixed, 48 items": lambda rng: np.full(T, 48),
               "mildly varied, 18-30": lambda rng: rng.integers(18, 31, size=T),
               "widely varied, 4-40": lambda rng: rng.integers(4, 41, size=T)}
    f_lim = (DIM * (PHASE_ONE + 1) * (PHASE_ONE - 1) / (PHASE_ONE * (PHASE_ONE - DIM))) * stats.f.ppf(
        1 - NOMINAL / 1000, DIM, PHASE_ONE - DIM)
    out = {"nominal_per_1000": NOMINAL, "streams": STREAMS, "phase_one": PHASE_ONE, "rows": [],
           "by_size": []}
    print(f"{STREAMS} streams per design, Phase I {PHASE_ONE}, F limit {f_lim:.2f}\n")
    print(f"{'design':>22}{'exposure':>10}{'rate':>8}{'95% CI':>16}{'q10':>7}{'median':>8}{'q90':>7}{'>2x nominal':>13}")
    for name, sizes_fn in designs.items():
        rng_master = np.random.default_rng(777)
        alarms = exposure = 0
        cond = []
        size_alarm = {}
        for s in range(STREAMS):
            rng = np.random.default_rng(20000 + s)
            sizes = sizes_fn(rng)
            b = rng.multivariate_normal(np.zeros(DIM), B, size=T)
            counts = np.array([rng.multinomial(int(n), P) for n in sizes], dtype=float)
            z = ilr(bayesian_multiplicative_replacement(counts)) + b
            mu = z[:PHASE_ONE].mean(axis=0)
            inv = np.linalg.pinv(np.cov(z[:PHASE_ONE], rowvar=False))
            d = z - mu
            stat = np.einsum("ij,jk,ik->i", d, inv, d)[PHASE_ONE:][WARMUP:]
            sz = sizes[PHASE_ONE:][WARMUP:]
            al = stat > f_lim
            alarms += int(al.sum()); exposure += len(al)
            cond.append(1000.0 * al.mean())
            for lo, hi, lab in ((4, 12, "4-12"), (13, 24, "13-24"), (25, 40, "25-40")):
                m = (sz >= lo) & (sz <= hi)
                a, n = size_alarm.get(lab, (0, 0))
                size_alarm[lab] = (a + int(al[m].sum()), n + int(m.sum()))
        rate = 1000.0 * alarms / exposure
        lo = stats.beta.ppf(0.025, alarms, exposure - alarms + 1) * 1000 if alarms else 0.0
        hi = stats.beta.ppf(0.975, alarms + 1, exposure - alarms) * 1000
        cond = np.array(cond)
        out["rows"].append({"design": name, "exposure": exposure, "alarms": alarms, "rate": rate,
                            "ci": [lo, hi], "q10": float(np.quantile(cond, .1)),
                            "median": float(np.quantile(cond, .5)), "q90": float(np.quantile(cond, .9)),
                            "share_above_2x": float(np.mean(cond > 2 * NOMINAL)),
                            "share_silent": float(np.mean(cond == 0))})
        for lab, (a, n) in size_alarm.items():
            if n:
                out["by_size"].append({"design": name, "size_class": lab, "alarms": a, "units": n,
                                       "rate": 1000.0 * a / n})
        print(f"{name:>22}{exposure:>10}{rate:>8.2f}{f'[{lo:.2f}, {hi:.2f}]':>16}"
              f"{np.quantile(cond,.1):>7.1f}{np.quantile(cond,.5):>8.1f}{np.quantile(cond,.9):>7.1f}"
              f"{100*np.mean(cond>2*NOMINAL):>12.0f}%")
    print("\nrate by unit-size class, per 1,000:")
    for r in out["by_size"]:
        if "varied" in r["design"]:
            print(f"  {r['design']:>22} {r['size_class']:>6}: {r['rate']:6.2f}  ({r['units']:,} units)")
    RESULTS.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS / "plain_f_limit_study.json", "w"), indent=2)
    print("\nwrote results/plain_f_limit_study.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
