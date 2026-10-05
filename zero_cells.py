#!/usr/bin/env python3
"""How much of the closed form's error at small counts is zero replacement?

The closed-form sampling covariance is asymptotic. Its maximum entry-wise relative
error against Monte Carlo is reported at four (composition, n) settings, once over
all multinomial draws and once restricted to draws with no empty cell, which
isolates the part attributable to replacing zeros from the part attributable to the
first-order delta method. Note that the zero-free column is not a valid decomposition of the error: conditioning
on the absence of an empty cell changes the sampling distribution, so the restricted
sample is compared with a closed form derived for the unrestricted one. It is reported
to show that, and not as an estimate of the effect of zero replacement.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wmmon.composition import bayesian_multiplicative_replacement, ilr, ilr_basis
from wmmon.laney_coda import sampling_covariance
from wmmon.paths import WMMON_HOME

RESULTS = Path(f"{WMMON_HOME}/results")
DRAWS = 400_000
SETTINGS = [([.5, .2, .15, .1, .05], 25), ([.5, .2, .15, .1, .05], 200),
            ([.5, .2, .15, .1, .05], 400), ([.5, .2, .15, .1, .05], 2000),
            ([.70, .10, .08, .07, .05], 50), ([.4, .3, .2, .06, .04], 100)]


def worst_entry(z, theory):
    emp = np.cov(z, rowvar=False)
    rel = np.abs(emp - theory) / np.sqrt(np.outer(np.diag(theory), np.diag(theory)))
    return float(rel.max())


def main() -> int:
    rng = np.random.default_rng(0)
    rows = []
    print(f"{'composition':<32}{'n':>5}{'empty-cell draws':>18}{'all draws':>11}{'zero-free':>11}")
    for p, n in SETTINGS:
        p = np.array(p)
        V = ilr_basis(len(p))
        theory = sampling_covariance(p, float(n), V)
        counts = rng.multinomial(n, p, size=DRAWS).astype(float)
        has_zero = (counts == 0).any(axis=1)
        z_all = ilr(bayesian_multiplicative_replacement(counts))
        e_all = worst_entry(z_all, theory)
        e_free = worst_entry(z_all[~has_zero], theory) if (~has_zero).sum() > 1000 else float("nan")
        rows.append({"p": p.tolist(), "n": n, "share_empty": float(has_zero.mean()),
                     "error_all": e_all, "error_zero_free": e_free})
        print(f"{str(p.tolist()):<32}{n:>5}{100*has_zero.mean():>17.1f}%{e_all:>11.3f}{e_free:>11.3f}")
    RESULTS.mkdir(exist_ok=True)
    json.dump({"draws": DRAWS, "rows": rows}, open(RESULTS / "zero_cells.json", "w"), indent=2)
    print("\nwrote results/zero_cells.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
