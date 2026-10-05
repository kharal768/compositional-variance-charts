#!/usr/bin/env python3
"""Is the closed form's error at small units isotropic, or concentrated?

Section 4.3 reports the closed-form sampling covariance as about 41% out
entry-wise at 25 items per unit, as a single worst-entry number. That number
says nothing about shape, yet the shape-tolerance check perturbs the covariance
anisotropically on the assumption that the real error might be anisotropic too.
This measures the error per balance, so the assumption can be checked rather
than asserted.

Multinomial draws are simulated, log-ratio coordinates formed with the same
zero-replacement the chart uses, and the empirical covariance compared with the
closed form. Reported per balance: the ratio of empirical to theoretical
variance. A flat profile means the error is a scale error; a varying profile
means it has shape, and the anisotropic perturbation test is measuring something
real.

Two compositions are used: the synthetic one behind the figure in Section 4.3,
and the class proportions of the labelled WM-811K subset, which are far sparser.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wmmon.composition import bayesian_multiplicative_replacement, ilr, ilr_basis
from wmmon.laney_coda import sampling_covariance
from wmmon.paths import WMMON_HOME

RESULTS = Path(f"{WMMON_HOME}/results")
DRAWS = 200_000


def profile(p, n, draws=DRAWS, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, p, size=draws).astype(float)
    closed = bayesian_multiplicative_replacement(counts)
    coords = ilr(closed)
    emp = np.cov(coords, rowvar=False)
    V = ilr_basis(len(p))
    theory = sampling_covariance(p, float(n), V)
    ratio = np.diag(emp) / np.diag(theory)
    off = np.abs(emp - theory)[~np.eye(len(ratio), dtype=bool)].max()
    return ratio, float(np.abs(emp / theory - 1).max()), off, np.trace(emp) / np.trace(theory)


def main() -> int:
    census = json.load(open(RESULTS / "census.json"))["label_counts"]
    order = ["none", "Edge-Ring", "Edge-Loc", "Center", "Loc", "Scratch",
             "Random", "Donut", "Near-full"]
    wafer = np.array([census[c] for c in order], dtype=float)
    wafer /= wafer.sum()
    synthetic = np.array([0.05, 0.10, 0.15, 0.20, 0.50])

    out = {"draws": DRAWS, "cases": []}
    for name, p in (("synthetic, min part 0.05", synthetic),
                    ("WM-811K class proportions", wafer)):
        print(f"\n{name}  (D = {len(p)}, smallest part {p.min():.4f})")
        print(f"{'items/unit':>11}{'trace ratio':>13}{'worst entry':>13}   per-balance variance ratio")
        for n in (25, 50, 200):
            ratio, worst, off, tr = profile(p, n)
            spread = float(ratio.max() / ratio.min())
            out["cases"].append({"composition": name, "n": n,
                                 "trace_ratio": float(tr), "worst_entry_error": worst,
                                 "per_balance_ratio": [float(x) for x in ratio],
                                 "ratio_spread": spread})
            print(f"{n:>11}{tr:>13.3f}{worst:>13.3f}   "
                  + " ".join(f"{x:5.2f}" for x in ratio) + f"   spread {spread:.2f}x")
    json.dump(out, open(RESULTS / "covariance_error_shape.json", "w"), indent=2)
    print("\nwrote results/covariance_error_shape.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
