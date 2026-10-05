#!/usr/bin/env python3
"""A null distribution for the batch-structure test, not a single permutation.

The mechanism test destroys batch structure by reassigning wafers to lot slots
at random and asks whether the uncorrected chart returns to its nominal rate.
Run once, that is a single draw. This repeats the permutation and reports the
distribution of realised rates, so the lot-structured rate can be placed
against it and a permutation p-value computed.

It also addresses the power question. With a few hundred evaluated units and a
nominal rate of 5 per 1,000, the expected alarm count is small and exact
intervals are wide; reporting the spread across permutations shows how much of
the observed difference could be sampling variation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon import composition, laney_coda as L
from wmmon.paths import WMMON_HOME

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
NOMINAL = 5.0
N_PERM = int(__import__("os").environ.get("N_PERM", 40))


def rate(proba, unit, seed):
    n_units = len(proba) // unit
    p = proba[: n_units * unit]
    ids = np.array([f"u{i // unit:06d}" for i in range(len(p))], dtype=object)
    seq = [f"u{i:06d}" for i in range(n_units)]
    fr = composition.build_stream(ids, p, seq, mode="hard")
    C = composition.ilr_matrix(fr); P = composition.proportion_matrix(fr)
    S = fr["lot_size"].to_numpy()
    ph = max(20 * C.shape[1], int(0.4 * n_units)); warm = 20
    exposure = n_units - ph - warm
    fit = L.calibrate(C[:ph], P[:ph], S[:ph], target_far=NOMINAL / 1000.0, seed=0)
    k = int(L.run_uncorrected(fit, C[ph:], P[ph:], S[ph:])["alarm"][warm:].sum())
    return 1000.0 * k / exposure, k, exposure


def main() -> int:
    proba = np.load(CACHE / "cnnproba_clean_1200_perm.npz")["p"]
    out = {"nominal_per_1000": NOMINAL, "n_permutations": N_PERM, "units": {}}
    for unit in (24, 48):
        observed, k_obs, exposure = rate(proba, unit, 0)
        null = []
        for s in range(N_PERM):
            shuffled = proba[np.random.default_rng(s).permutation(len(proba))]
            null.append(rate(shuffled, unit, s)[0])
        null = np.array(null)
        # one-sided: how often does a stream without batch structure reach the
        # rate the lot-structured stream reached?
        p_value = (1 + int((null >= observed).sum())) / (1 + len(null))
        lo = stats.beta.ppf(0.025, k_obs, exposure - k_obs + 1) * 1000 if k_obs else 0.0
        hi = stats.beta.ppf(0.975, k_obs + 1, exposure - k_obs) * 1000
        out["units"][str(unit)] = {
            "observed_rate": observed, "observed_ci": [lo, hi],
            "exposure_units": exposure, "expected_alarms_at_nominal": exposure * NOMINAL / 1000,
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_min": float(null.min()), "null_max": float(null.max()),
            "null_q95": float(np.quantile(null, 0.95)), "p_value": p_value,
        }
        r = out["units"][str(unit)]
        print(f"unit {unit}: observed {observed:.1f} [{lo:.1f}, {hi:.1f}] | "
              f"null over {N_PERM} permutations: mean {null.mean():.1f}, "
              f"sd {null.std(ddof=1):.1f}, max {null.max():.1f} | p = {p_value:.3f}")
        print(f"          exposure {exposure} units, {exposure * NOMINAL / 1000:.1f} "
              f"alarms expected at nominal")
    RESULTS.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS / "permutation_null.json", "w"), indent=2)
    print("\nwrote results/permutation_null.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
