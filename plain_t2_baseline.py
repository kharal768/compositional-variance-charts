#!/usr/bin/env python3
"""Does the components-of-variance machinery beat a plain Phase I covariance?

At fixed unit size the total covariance Sigma_samp + Sigma_b is the same for
every unit, so an individuals Hotelling T^2 on log-ratio coordinates, using the
Phase I sample covariance and an empirical limit, should calibrate just as well
without any decomposition. If so, the machinery earns its place only where unit
sizes vary, and the paper should say that.

Three streams are simulated, all in control, all with genuine between-unit
variation:

  fixed          every unit the same size
  mildly varied  sizes 18-30, as lots of a nominal 25 vary in practice
  widely varied  sizes 4-40, as a stream including short lots

Both charts are calibrated on the same Phase I window to the same nominal rate
and scored on the same Phase II units, so the only difference is whether the
reference covariance is decomposed and rebuilt per unit or estimated once.
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
NOMINAL, T = 5.0, 1600


def plain_t2_f(coords, phase_one, warmup, nominal):
    """The same statistic with the textbook Phase II limit for estimated mean and covariance."""
    ref = coords[:phase_one]
    d_ = coords.shape[1]
    inv = np.linalg.pinv(np.cov(ref, rowvar=False))
    d = coords - ref.mean(axis=0)
    stat = np.einsum("ij,jk,ik->i", d, inv, d)
    m = phase_one
    limit = (d_ * (m + 1) * (m - 1) / (m * (m - d_))) * stats.f.ppf(1 - nominal / 1000.0, d_, m - d_)
    return int((stat[phase_one:][warmup:] > limit).sum())


def plain_t2(coords, sizes, phase_one, warmup, nominal, seed=0):
    """Individuals T^2 on the Phase I sample covariance, empirical limit."""
    ref = coords[:phase_one]
    mu = ref.mean(axis=0)
    S = np.cov(ref, rowvar=False)
    inv = np.linalg.pinv(S)
    d = coords - mu
    stat = np.einsum("ij,jk,ik->i", d, inv, d)
    limit = float(np.quantile(stat[:phase_one], 1.0 - nominal / 1000.0))
    test = stat[phase_one:]
    return int((test[warmup:] > limit).sum())


def stream(sizes, p, B, seed=0):
    rng = np.random.default_rng(seed)
    dim = len(p) - 1
    b = rng.multivariate_normal(np.zeros(dim), B, size=len(sizes))
    counts = np.array([rng.multinomial(int(n), p) for n in sizes], dtype=float)
    return ilr(bayesian_multiplicative_replacement(counts)) + b, counts


def interval(k, n):
    lo = stats.beta.ppf(0.025, k, n - k + 1) * 1000 if k else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) * 1000 if k < n else 1000.0
    return lo, hi


def main() -> int:
    rng = np.random.default_rng(0)
    p = np.array([0.808, 0.053, 0.028, 0.024, 0.020, 0.017, 0.017, 0.017, 0.017])
    p = p / p.sum()
    V = ilr_basis(len(p))
    dim = len(p) - 1
    base = sampling_covariance(p, 25.0, V)
    B = np.diag(np.linspace(1.0, 0.3, dim))
    B *= 0.30 * np.trace(base) / np.trace(B)

    designs = {
        "fixed, 25 items": np.full(T, 25.0),
        "mildly varied, 18-30": rng.integers(18, 31, size=T).astype(float),
        "widely varied, 4-40": rng.integers(4, 41, size=T).astype(float),
    }
    phase_one, warmup = int(0.4 * T), 20
    exposure = T - phase_one - warmup
    rows = []
    print(f"{T} units, nominal {NOMINAL:.1f} per 1,000, exposure {exposure}\n")
    print(f"{'unit sizes':>22}{'plain T2':>11}{'covers':>8}"
          f"{'variance components':>21}{'covers':>8}")
    for label, sizes in designs.items():
        coords, counts = stream(sizes, p, B)
        props = bayesian_multiplicative_replacement(counts)
        k_plain = plain_t2(coords, sizes, phase_one, warmup, NOMINAL)
        k_f = plain_t2_f(coords, phase_one, warmup, NOMINAL)
        lf, hf = interval(k_f, exposure)
        fit = VC.calibrate(coords[:phase_one], props[:phase_one], sizes[:phase_one],
                           target_far=NOMINAL / 1000.0, estimator="em", seed=0)
        out = VC.run(fit, coords[phase_one:], props[phase_one:], sizes[phase_one:])
        k_vc = int(out["alarm"][warmup:].sum())
        lp, hp = interval(k_plain, exposure)
        lv, hv = interval(k_vc, exposure)
        rows.append({"design": label,
                     "plain_rate": 1000.0 * k_plain / exposure, "plain_ci": [lp, hp],
                     "plain_covers": bool(lp <= NOMINAL <= hp),
                     "vc_rate": 1000.0 * k_vc / exposure, "vc_ci": [lv, hv],
                     "vc_covers": bool(lv <= NOMINAL <= hv),
                     "plain_f_rate": 1000.0 * k_f / exposure, "plain_f_ci": [lf, hf],
                     "plain_f_covers": bool(lf <= NOMINAL <= hf)})
        print(f"{label:>22}  F-limit plain: {1000.0*k_f/exposure:>6.1f} [{lf:.1f}, {hf:.1f}]")
        print(f"{label:>22}{1000.0*k_plain/exposure:>11.1f}"
              f"{'yes' if lp <= NOMINAL <= hp else 'NO':>8}"
              f"{1000.0*k_vc/exposure:>21.1f}"
              f"{'yes' if lv <= NOMINAL <= hv else 'NO':>8}")
    RESULTS.mkdir(exist_ok=True)
    json.dump({"nominal_per_1000": NOMINAL, "T": T, "exposure": exposure, "rows": rows},
              open(RESULTS / "plain_t2_baseline.json", "w"), indent=2)
    print("\nwrote results/plain_t2_baseline.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
