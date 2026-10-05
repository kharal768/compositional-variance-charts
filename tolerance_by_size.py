#!/usr/bin/env python3
"""Misspecification of the known term, crossed with the dispersion of unit sizes.

At a fixed unit size the total covariance is one matrix, so a scale error in the assumed
sampling term is taken up exactly by the between-unit estimate for as long as that estimate
stays positive semi-definite. With varying sizes the sampling term scales as 1/n_t while the
between-unit term does not, so a scale error cannot be absorbed by any constant matrix. This
measures the realised in-control rate against the scale factor k applied to the closed form,
for a fixed size and for sizes 4 to 40, on streams with the same between-unit covariance.
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
NOMINAL, T, STREAMS = 5.0, 1600, 2
FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)


def main() -> int:
    p = np.array([0.808, 0.053, 0.028, 0.024, 0.020, 0.017, 0.017, 0.017, 0.017])
    p = p / p.sum()
    V = ilr_basis(len(p)); dim = len(p) - 1
    base = sampling_covariance(p, 25.0, V)
    B = np.diag(np.linspace(1.0, 0.3, dim)); B *= 0.30 * np.trace(base) / np.trace(B)
    designs = {"fixed, 25 items": lambda rng: np.full(T, 25),
               "widely varied, 4-40": lambda rng: rng.integers(4, 41, size=T)}
    phase_one, warmup = int(0.4 * T), 20
    exposure = STREAMS * (T - phase_one - warmup)
    rows = []
    print(f"exposure {exposure} units per cell, nominal {NOMINAL:.1f} per 1,000\n")
    print(f"{'unit sizes':>22}" + "".join(f"{f'k={k:g}':>10}" for k in FACTORS))
    for name, sizes_fn in designs.items():
        counts_by_k = {k: 0 for k in FACTORS}
        for s in range(STREAMS):
            rng = np.random.default_rng(3000 + s)
            sizes = sizes_fn(rng).astype(float)
            b = rng.multivariate_normal(np.zeros(dim), B, size=T)
            counts = np.array([rng.multinomial(int(n), p) for n in sizes], dtype=float)
            props = bayesian_multiplicative_replacement(counts)
            coords = ilr(props) + b
            for k in FACTORS:
                fit = VC.calibrate(coords[:phase_one], props[:phase_one], sizes[:phase_one],
                                   sampling_scale=k, target_far=NOMINAL / 1000.0,
                                   estimator="em", seed=s, n_bootstrap=6000)
                out = VC.run(fit, coords[phase_one:], props[phase_one:], sizes[phase_one:])
                counts_by_k[k] += int(out["alarm"][warmup:].sum())
        line = f"{name:>22}"
        for k in FACTORS:
            c = counts_by_k[k]
            lo = stats.beta.ppf(0.025, c, exposure - c + 1) * 1000 if c else 0.0
            hi = stats.beta.ppf(0.975, c + 1, exposure - c) * 1000
            rows.append({"design": name, "k": k, "alarms": c, "rate": 1000.0 * c / exposure,
                         "ci": [lo, hi], "exposure": exposure})
            line += f"{1000.0 * c / exposure:>10.1f}"
        print(line)
    RESULTS.mkdir(exist_ok=True)
    json.dump({"nominal_per_1000": NOMINAL, "rows": rows}, open(RESULTS / "tolerance_by_size.json", "w"), indent=2)
    print("\nwrote results/tolerance_by_size.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
