#!/usr/bin/env python3
"""Do the trace-ratio thresholds hold away from the configuration that produced them?

Section 6.8 gives two rules - a trace ratio near 0.2 for the control limit, near
0.4 for a reported decomposition - measured at one dimension with one diagonal
between-unit covariance. This varies both: three dimensions and three shapes of
Sigma_b, including a non-diagonal one and one with a wide eigenvalue spread. If
the thresholds are properties of the estimator they should move little; if they
are properties of that configuration they will move a lot.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wmmon import varcomp as VC
from wmmon.composition import ilr_basis
from wmmon.laney_coda import sampling_covariance
from wmmon.paths import WMMON_HOME

RESULTS = Path(f"{WMMON_HOME}/results")
T = 900


def between(dim, shape, rng):
    if shape == "diagonal, mild":
        d = np.linspace(0.02, 0.002, dim)
        return np.diag(d)
    if shape == "diagonal, wide spread":
        d = 0.02 * np.logspace(0, -2.5, dim)
        return np.diag(d)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    d = np.linspace(0.02, 0.004, dim)
    return Q @ np.diag(d) @ Q.T


def recovery(dim, shape, n_unit, rng):
    p = np.full(dim + 1, 1.0 / (dim + 1))
    p[0] = 0.5
    p[1:] = 0.5 / dim
    V = ilr_basis(dim + 1)
    S = sampling_covariance(p, float(n_unit), V)
    B = between(dim, shape, rng)
    ratio = float(np.trace(B) / np.trace(S))
    b = rng.multivariate_normal(np.zeros(dim), B, size=T)
    e = rng.multivariate_normal(np.zeros(dim), S, size=T)
    est, _, _ = VC.fit_between_em(b + e, np.zeros(dim), np.array([S] * T),
                                  iterations=250)
    total = abs(np.trace(est) - np.trace(B)) / np.trace(B)
    comp = float(np.mean(np.abs(np.diag(est) - np.diag(B)) / np.diag(B)))
    return ratio, float(total), comp


def main() -> int:
    rows = []
    print(f"{'dim':>4}{'Sigma_b shape':>24}{'trace ratio':>13}"
          f"{'total err':>11}{'component err':>15}")
    for dim in (4, 8):
        for shape in ("diagonal, mild", "diagonal, wide spread", "non-diagonal"):
            rng = np.random.default_rng(0)
            for n_unit in (50, 100, 200, 400):
                ratio, total, comp = recovery(dim, shape, n_unit, rng)
                rows.append({"dim": dim, "shape": shape, "n_unit": n_unit,
                             "trace_ratio": ratio, "total_rel_error": total,
                             "component_rel_error": comp})
            # the ratio at which the total error first falls below 0.15
            ok = [r for r in rows if r["dim"] == dim and r["shape"] == shape
                  and r["total_rel_error"] < 0.15]
            thr = min((r["trace_ratio"] for r in ok), default=float("nan"))
            sel = [r for r in rows if r["dim"] == dim and r["shape"] == shape]
            print(f"{dim:>4}{shape:>24}{'':>13}{'':>11}{'':>15}")
            for r in sel:
                print(f"{'':>4}{'':>24}{r['trace_ratio']:>13.3f}"
                      f"{r['total_rel_error']:>11.3f}{r['component_rel_error']:>15.3f}")
            print(f"{'':>4}{'threshold (total err < 0.15)':>24}{thr:>13.3f}")
    json.dump({"iterations": 400, "T": T, "rows": rows},
              open(RESULTS / "threshold_generality.json", "w"), indent=2)
    print("\nwrote results/threshold_generality.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
