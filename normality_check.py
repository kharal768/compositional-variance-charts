#!/usr/bin/env python3
"""Is the charting statistic's reference distribution right at these unit sizes?

The control limit is obtained by simulating whitened vectors from a normal with
the fitted covariance. At 24 items per unit the log-ratio coordinates come from
sparse, zero-replaced counts, which is where Kartiosuo et al. report the normal
approximation degrading. If the simulated reference is wrong in the tail, the
limit is wrong, whatever the point estimates say.

This compares the simulated reference distribution of the statistic against the
empirical distribution from in-control units, at the quantiles a control chart
actually uses, and reports a Kolmogorov-Smirnov distance and the realised
exceedance at each nominal level.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon import composition, varcomp as VC
from wmmon.laney_coda import sampling_covariance
from wmmon.paths import WMMON_HOME

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
UNIT = 24


def main() -> int:
    proba = np.load(CACHE / "cnnproba_clean_1200_perm.npz")["p"]
    n_units = len(proba) // UNIT
    p = proba[: n_units * UNIT]
    ids = np.array([f"u{i // UNIT:06d}" for i in range(len(p))], dtype=object)
    seq = [f"u{i:06d}" for i in range(n_units)]
    fr = composition.build_stream(ids, p, seq, mode="hard")
    coords = composition.ilr_matrix(fr)
    props = composition.proportion_matrix(fr)
    sizes = fr["lot_size"].to_numpy()
    phase_one = max(160, int(0.4 * n_units))

    fit = VC.calibrate(coords[:phase_one], props[:phase_one], sizes[:phase_one],
                       target_far=5.0 / 1000.0, estimator="em", lag=4, seed=0)
    # empirical statistic on the held-out in-control units
    out = VC.run(fit, coords[phase_one:], props[phase_one:], sizes[phase_one:])
    empirical = np.asarray(out["statistic"])[20:]

    # simulated reference, the distribution the limit is read from
    rng = np.random.default_rng(0)
    cov = sampling_covariance(fit.reference, float(UNIT), fit.basis) + fit.between
    root = np.linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0]))
    inv = np.linalg.pinv(cov)
    draws = rng.standard_normal((200_000, cov.shape[0])) @ root.T
    simulated = np.einsum("ij,jk,ik->i", draws, inv, draws)

    ks = stats.ks_2samp(empirical, simulated)
    print(f"units evaluated {len(empirical)} | KS distance {ks.statistic:.3f}, "
          f"p = {ks.pvalue:.3f}\n")
    print(f"{'nominal':>9}{'simulated limit':>17}{'empirical quantile':>20}"
          f"{'exceedance/1,000':>18}")
    rows = []
    for nominal in (50.0, 20.0, 10.0, 5.0):
        q = 1.0 - nominal / 1000.0
        lim = float(np.quantile(simulated, q))
        emp_q = float(np.quantile(empirical, q))
        exceed = 1000.0 * float((empirical > lim).mean())
        rows.append({"nominal_per_1000": nominal, "simulated_limit": lim,
                     "empirical_quantile": emp_q, "realised_per_1000": exceed})
        print(f"{nominal:>9.0f}{lim:>17.2f}{emp_q:>20.2f}{exceed:>18.1f}")
    # chi-square reference for comparison: what the asymptotic limit would give
    chi = float(stats.chi2.ppf(1 - 5.0 / 1000.0, cov.shape[0]))
    chi_rate = 1000.0 * float((empirical > chi).mean())
    print(f"\nasymptotic chi-square limit at nominal 5.0: {chi:.2f} "
          f"-> realised {chi_rate:.1f} per 1,000")
    RESULTS.mkdir(exist_ok=True)
    json.dump({"unit": UNIT, "n_evaluated": int(len(empirical)),
               "ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
               "levels": rows, "chi2_limit": chi, "chi2_realised": chi_rate},
              open(RESULTS / "normality_check.json", "w"), indent=2)
    print("\nwrote results/normality_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
