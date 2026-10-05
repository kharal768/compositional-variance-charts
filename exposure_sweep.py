#!/usr/bin/env python3
"""Calibration at an exposure the wafer stream cannot provide.

Every realised rate on WM-811K rests on a few hundred in-control units and one
estimated limit, so the exact interval reaches far past the nominal value and
the evidence cannot separate a calibrated chart from one that is several times
off. The simulated setting has no such limit.

Independent streams are generated, each with its own Phase I window and its own
estimated limit, and alarms are pooled across them. This gives an exposure two
orders of magnitude larger than the wafer stream and, because each stream draws
a fresh Phase I, the result is a marginal false-alarm rate rather than one
conditional on a single estimate - the distinction Li, Tsung & Zou show matters.

Reported at two unit sizes, for the chart and for the plain Phase I covariance
baseline of section 6.11.
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
NOMINAL = 5.0
STREAMS, T = 25, 600


def plain(coords, phase_one, warmup, nominal):
    ref = coords[:phase_one]
    inv = np.linalg.pinv(np.cov(ref, rowvar=False))
    d = coords - ref.mean(axis=0)
    stat = np.einsum("ij,jk,ik->i", d, inv, d)
    limit = float(np.quantile(stat[:phase_one], 1.0 - nominal / 1000.0))
    return int((stat[phase_one:][warmup:] > limit).sum())


def main() -> int:
    p = np.array([0.808, 0.053, 0.028, 0.024, 0.020, 0.017, 0.017, 0.017, 0.017])
    p = p / p.sum()
    V = ilr_basis(len(p))
    dim = len(p) - 1
    out = {"nominal_per_1000": NOMINAL, "streams": STREAMS, "units_per_stream": T,
           "rows": []}
    print(f"{STREAMS} independent streams x {T} units, nominal {NOMINAL:.1f} per 1,000\n")
    print(f"{'unit size':>10}{'method':>22}{'alarms':>8}{'exposure':>10}"
          f"{'rate':>8}{'95% CI':>16}{'covers':>8}")
    for n_unit in (24, 48):
        base = sampling_covariance(p, float(n_unit), V)
        B = np.diag(np.linspace(1.0, 0.3, dim))
        B *= 0.30 * np.trace(base) / np.trace(B)
        tot = {"variance components": 0, "plain Phase I covariance": 0}
        per_stream = {"variance components": [], "plain Phase I covariance": []}
        per_exposure = None
        exposure = 0
        for s in range(STREAMS):
            rng = np.random.default_rng(1000 + s)
            b = rng.multivariate_normal(np.zeros(dim), B, size=T)
            counts = np.array([rng.multinomial(n_unit, p) for _ in range(T)], dtype=float)
            props = bayesian_multiplicative_replacement(counts)
            coords = ilr(props) + b
            sizes = np.full(T, float(n_unit))
            phase_one, warmup = int(0.4 * T), 20
            exposure += T - phase_one - warmup
            fit = VC.calibrate(coords[:phase_one], props[:phase_one], sizes[:phase_one],
                               target_far=NOMINAL / 1000.0, estimator="em", seed=s)
            r = VC.run(fit, coords[phase_one:], props[phase_one:], sizes[phase_one:])
            k_vc = int(r["alarm"][warmup:].sum())
            k_pl = plain(coords, phase_one, warmup, NOMINAL)
            tot["variance components"] += k_vc
            tot["plain Phase I covariance"] += k_pl
            per_stream["variance components"].append(k_vc)
            per_stream["plain Phase I covariance"].append(k_pl)
            per_exposure = T - phase_one - warmup
        for method, k in tot.items():
            lo = stats.beta.ppf(0.025, k, exposure - k + 1) * 1000 if k else 0.0
            hi = stats.beta.ppf(0.975, k + 1, exposure - k) * 1000
            covers = lo <= NOMINAL <= hi
            cond = 1000.0 * np.array(per_stream[method]) / per_exposure
            out["rows"].append({"unit": n_unit, "method": method, "alarms": k,
                                "exposure": exposure, "rate": 1000.0 * k / exposure,
                                "ci": [lo, hi], "covers": bool(covers),
                                "ratio_to_nominal": (1000.0 * k / exposure) / NOMINAL,
                                "ratio_ci": [lo / NOMINAL, hi / NOMINAL],
                                "per_stream_rates": [float(x) for x in cond],
                                "conditional_q10": float(np.quantile(cond, 0.10)),
                                "conditional_median": float(np.quantile(cond, 0.50)),
                                "conditional_q90": float(np.quantile(cond, 0.90)),
                                "share_streams_silent": float((cond == 0).mean())})
            print(f"{n_unit:>10}{method:>22}{k:>8}{exposure:>10}"
                  f"{1000.0*k/exposure:>8.2f}{f'[{lo:.2f}, {hi:.2f}]':>16}"
                  f"{'yes' if covers else 'NO':>8}")
    RESULTS.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS / "exposure_sweep.json", "w"), indent=2)
    print("\nwrote results/exposure_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
