#!/usr/bin/env python3
"""Does cutting units across lot boundaries create the dependence the lag removes?

Fixed blocks of 24 wafers are cut irrespective of lot boundaries, and the sampled
stream averages about 17.5 wafers per lot, so adjacent blocks share lots by
construction. Shared membership induces correlation between neighbouring units on
its own, which is exactly what the lag correction removes - so the reported
clustering may be an artefact of how units were built rather than a property of
production.

The question is separable in simulation, where batch structure can be switched on
and off. Lots are generated with sizes drawn like production lots, each with its
own composition when batch structure is present and the common composition when it
is absent. Units are then built two ways from the same wafers:

  fixed blocks   a constant number of consecutive wafers, crossing lot boundaries
  lot-aligned    whole lots grouped to the same median size, never split

For each, the lag-1 correlation of the whitened coordinates and the dispersion
summary at lags 1 and 4 are reported. If block construction alone creates the
dependence, the no-batch-structure stream will show it under fixed blocks and not
under lot-aligned units.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wmmon.composition import bayesian_multiplicative_replacement, ilr, ilr_basis
from wmmon.laney_coda import sampling_covariance
from wmmon.paths import WMMON_HOME

RESULTS = Path(f"{WMMON_HOME}/results")
N_LOTS, TARGET = 1400, 24
CONCENTRATION = 40      # lower means more variation between lots
CONCENTRATIONS = (40, 10, 4)


def build(batch_structure, seed=0, concentration=None):
    rng = np.random.default_rng(seed)
    p = np.array([0.808, 0.053, 0.028, 0.024, 0.020, 0.017, 0.017, 0.017, 0.017])
    p = p / p.sum()
    sizes = np.where(rng.random(N_LOTS) < 0.47, 25,
                     rng.integers(1, 25, size=N_LOTS)).astype(int)
    wafers, lot_of = [], []
    for i, n in enumerate(sizes):
        # a lot's own composition, or the common one if there is no batch structure
        q = rng.dirichlet(p * (concentration or CONCENTRATION)) if batch_structure else p
        wafers.append(rng.multinomial(1, q, size=n))
        lot_of.append(np.full(n, i))
    return np.vstack(wafers).astype(float), np.concatenate(lot_of), sizes, p


def summarise(counts_per_unit, p):
    props = bayesian_multiplicative_replacement(counts_per_unit)
    z = ilr(props)
    V = ilr_basis(len(p))
    sizes = counts_per_unit.sum(axis=1)
    mu = z.mean(axis=0)
    w = np.array([np.linalg.inv(np.linalg.cholesky(
        sampling_covariance(p, float(n), V))) @ (z[t] - mu)
        for t, n in enumerate(sizes)])
    lag1 = float(np.mean([np.corrcoef(w[:-1, j], w[1:, j])[0, 1]
                          for j in range(w.shape[1])]))
    out = {}
    for lag in (1, 2, 4, 8):
        d = w[lag:] - w[:-lag]
        S = d.T @ d / (2 * len(d))
        out[f"sigma_Z_lag{lag}"] = float(np.sqrt(np.trace(S) / w.shape[1]))
    out["lag1_correlation"] = lag1
    return out


def units_fixed(wafers, target):
    n = len(wafers) // target
    return np.array([wafers[i * target:(i + 1) * target].sum(axis=0) for i in range(n)])


def units_lot_aligned(wafers, lot_of, sizes, target):
    """Group whole lots until the group reaches the target size."""
    groups, cur, tot = [], [], 0
    for i, n in enumerate(sizes):
        cur.append(i); tot += n
        if tot >= target:
            groups.append(cur); cur, tot = [], 0
    rows = []
    for g in groups:
        sel = np.isin(lot_of, g)
        rows.append(wafers[sel].sum(axis=0))
    return np.array(rows)


def main() -> int:
    out = {"target_unit": TARGET, "rows": []}
    print(f"{'conc.':>6}{'batch':>7}{'units':>14}{'lag-1 corr':>12}"
          f"{'s_Z lag1':>10}{'lag2':>8}{'lag4':>8}{'lag8':>8}{'lag2/lag1':>11}")
    cases = [(c, True) for c in CONCENTRATIONS] + [(None, False)]
    for conc, bs in cases:
        wafers, lot_of, sizes, p = build(bs, concentration=conc)
        for name, u in (("fixed blocks", units_fixed(wafers, TARGET)),
                        ("lot-aligned", units_lot_aligned(wafers, lot_of, sizes, TARGET))):
            s = summarise(u, p)
            out["rows"].append({"concentration": conc, "batch_structure": bs, "units": name,
                                "n_units": len(u), **s})
            jump = s["sigma_Z_lag2"] / s["sigma_Z_lag1"]
            print(f"{str(conc):>6}{str(bs):>7}{name:>14}{s['lag1_correlation']:>12.3f}"
                  f"{s['sigma_Z_lag1']:>10.3f}{s['sigma_Z_lag2']:>8.3f}{s['sigma_Z_lag4']:>8.3f}"
                  f"{s['sigma_Z_lag8']:>8.3f}{jump:>11.3f}")
    RESULTS.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS / "block_alignment.json", "w"), indent=2)
    print("\nwrote results/block_alignment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
