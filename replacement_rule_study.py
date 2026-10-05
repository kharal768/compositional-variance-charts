#!/usr/bin/env python3
"""Does the plain chart's calibration, and the size-class pattern, depend on how zeros are replaced?

Every composition here has empty cells, so the log-ratio transform needs a replacement rule, and all the
simulations use one: the Bayesian-multiplicative rule with a Jeffreys prior. The sweep of S4 and S5 varied
the prior for the components-of-variance chart only. This varies the rule for the plain chart, and
measures what the rule does to the tails of the coordinates.

Rules
  BM, prior p    zero cells get prior / (total + D * prior), the same value for every empty cell of a
                 unit and decreasing with the unit's total; non-zero cells are shrunk to compensate.
                 p = 0.5 is the Jeffreys prior used elsewhere.
  zeros -> 0.5   each zero count is set to 0.5 and the row is closed, as in the simulations of
                 Kartiosuo et al. (2025).

Reported for each rule: (a) the excess kurtosis and skewness of the rarest-balance coordinate and the
tail of T^2 on a pure multinomial stream; (b) the F-limit rate of the plain chart with Gaussian
between-unit variation, at a fixed size and with sizes 4 to 40, with the rate by size class; (c) the ratio
of the empirical sampling-variance trace to the closed form at 4, 24 and 40 items.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon.composition import bayesian_multiplicative_replacement, ilr, ilr_basis
from wmmon.laney_coda import sampling_covariance
from wmmon.paths import WMMON_HOME

P = np.array([0.808, 0.053, 0.028, 0.024, 0.020, 0.017, 0.017, 0.017, 0.017]); P = P / P.sum()
V = ilr_basis(len(P)); DIM = len(P) - 1
NOMINAL, STREAMS, T, PHASE_ONE, WARMUP = 5.0, 150, 600, 240, 20


def replace(rule, counts):
    if rule == "zeros -> 0.5":
        c = np.where(counts == 0, 0.5, counts)
        return c / c.sum(axis=1, keepdims=True)
    return bayesian_multiplicative_replacement(counts, prior=rule)


RULES = [("BM, prior 1.0", 1.0), ("BM, prior 0.5 (Jeffreys)", 0.5), ("BM, prior 0.1", 0.1),
         ("BM, prior 0.05", 0.05), ("zeros -> 0.5", "zeros -> 0.5")]


def stream_rate(rule, sizes_fn, bscale, B0, seed0=40000):
    f_lim = (DIM * (PHASE_ONE + 1) * (PHASE_ONE - 1) / (PHASE_ONE * (PHASE_ONE - DIM))) * stats.f.ppf(
        1 - NOMINAL / 1000, DIM, PHASE_ONE - DIM)
    alarms = expo = 0
    by = {"4-12": [0, 0], "13-24": [0, 0], "25-40": [0, 0]}
    for s in range(STREAMS):
        rng = np.random.default_rng(seed0 + s)
        sizes = sizes_fn(rng)
        counts = np.array([rng.multinomial(int(n), P) for n in sizes], dtype=float)
        z = ilr(replace(rule, counts))
        if bscale > 0:
            z = z + rng.multivariate_normal(np.zeros(DIM), bscale * B0, size=T)
        mu = z[:PHASE_ONE].mean(axis=0)
        inv = np.linalg.pinv(np.cov(z[:PHASE_ONE], rowvar=False))
        d = z - mu
        stat = np.einsum("ij,jk,ik->i", d, inv, d)[PHASE_ONE:][WARMUP:]
        sz = sizes[PHASE_ONE:][WARMUP:]
        al = stat > f_lim
        alarms += int(al.sum()); expo += len(al)
        for lab, lo, hi in (("4-12", 4, 12), ("13-24", 13, 24), ("25-40", 25, 40)):
            m = (sz >= lo) & (sz <= hi); by[lab][0] += int(al[m].sum()); by[lab][1] += int(m.sum())
    return 1000.0 * alarms / expo, {k: (1000.0 * a / n if n else None) for k, (a, n) in by.items()}


def main() -> int:
    rng = np.random.default_rng(11)
    base = sampling_covariance(P, 25.0, V)
    B0 = np.diag(np.linspace(1.0, 0.3, DIM)); B0 *= 0.30 * np.trace(base) / np.trace(B0)
    fixed = lambda rng: np.full(T, 24)
    wide = lambda rng: rng.integers(4, 41, size=T)
    out = {"rows": []}
    print(f"{'rule':<26}{'kurt (rare)':>12}{'skew (rare)':>12}{'tail, n=24, B=0':>17}{'fixed 24':>10}{'fixed, B=0':>12}{'wide 4-40':>11}"
          f"{'  by size 4-12 / 13-24 / 25-40':>32}{'  trace ratio 4/24/40':>24}")
    for name, rule in RULES:
        counts = rng.multinomial(24, P, size=60000).astype(float)
        zr = ilr(replace(rule, counts))
        rare = zr[:, -1]
        t2 = np.einsum("ij,jk,ik->i", zr - zr.mean(0), np.linalg.inv(np.cov(zr, rowvar=False)), zr - zr.mean(0))
        tail = 1000.0 * float((t2 > stats.chi2.ppf(1 - NOMINAL / 1000, DIM)).mean())
        tr = {}
        for n in (4, 24, 40):
            c = rng.multinomial(n, P, size=60000).astype(float)
            tr[n] = float(np.trace(np.cov(ilr(replace(rule, c)), rowvar=False)) / np.trace(sampling_covariance(P, float(n), V)))
        r_fix, _ = stream_rate(rule, fixed, 1.0, B0)
        r_fix0, _ = stream_rate(rule, fixed, 0.0, B0)
        r_wide, by = stream_rate(rule, wide, 1.0, B0)
        row = {"rule": name, "kurtosis_rare": float(stats.kurtosis(rare)), "skew_rare": float(stats.skew(rare)),
               "tail_pure_multinomial": tail, "rate_fixed": r_fix, "rate_fixed_no_between": r_fix0,
               "rate_wide": r_wide, "by_size": by, "trace_ratio": tr}
        out["rows"].append(row)
        print(f"{name:<26}{row['kurtosis_rare']:>12.2f}{row['skew_rare']:>12.2f}{tail:>17.2f}{r_fix:>10.2f}{r_fix0:>12.2f}{r_wide:>11.2f}"
              f"{by['4-12']:>12.2f}/{by['13-24']:>6.2f}/{by['25-40']:>6.2f}{tr[4]:>10.3f}/{tr[24]:.2f}/{tr[40]:.2f}", flush=True)
    Path(f"{WMMON_HOME}/results").mkdir(exist_ok=True)
    json.dump(out, open(Path(f"{WMMON_HOME}/results")/"replacement_rule_study.json", "w"), indent=2)
    print("\nwrote results/replacement_rule_study.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
