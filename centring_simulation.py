#!/usr/bin/env python3
"""The centring correction on a simulated stream, where the truth is known.

The result on real lots rests on one dataset and one inferred ordering. Here the
same comparison is run on a stream generated from a fixed composition with no
batch structure at all, so any correlation between unit size and the charting
statistic is an artefact of centring and nothing else. Unit sizes follow the
distribution of real production lots: most at the nominal size, a tail of short
ones.

Three centrings are compared: a single Phase I mean, the first-order correction,
and the first plus second order term of E[log p-hat]. With no batch structure
present the correct answer is no correlation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon import composition, varcomp as VC
from wmmon.composition import bayesian_multiplicative_replacement, ilr, ilr_basis
from wmmon.paths import WMMON_HOME

RESULTS = Path(f"{WMMON_HOME}/results")
NOMINAL = 5.0
T = 1200


def offsets(reference, sizes, basis):
    """First-order term, the third-moment second-order term, and the fourth-moment one.

    E[log p-hat] = log p - (1-p)/(2Kp) + (1-p)(1-2p)/(3K^2 p^2) - 3(1-p)^2/(4K^2 p^2)
                   + O(K^-3).
    The last two terms are both order K^-2. An earlier version of this script used only
    the third-moment term, which against the exact binomial expectation is less accurate
    than the first-order term alone.
    """
    p = np.clip(reference, 1e-12, None)
    first = -np.outer(1.0 / sizes, (1.0 - p) / (2.0 * p)) @ basis.T
    third = np.outer(1.0 / sizes ** 2,
                     (1.0 - p) * (1.0 - 2.0 * p) / (3.0 * p ** 2)) @ basis.T
    fourth = -np.outer(1.0 / sizes ** 2, 3.0 * (1.0 - p) ** 2 / (4.0 * p ** 2)) @ basis.T
    return first, third, fourth


def main() -> int:
    rng = np.random.default_rng(0)
    # a composition like the monitored one: one dominant class, a sparse tail
    p = np.array([0.808, 0.053, 0.028, 0.024, 0.020, 0.017, 0.017, 0.017, 0.017])
    p = p / p.sum()
    # lot sizes as in production: nominal 25, a tail of short lots
    sizes = np.where(rng.random(T) < 0.55, 25,
                     rng.integers(1, 25, size=T)).astype(float)
    counts = np.array([rng.multinomial(int(n), p) for n in sizes], dtype=float)
    closed = bayesian_multiplicative_replacement(counts)
    coords = ilr(closed)
    basis = ilr_basis(len(p))
    phase_one = int(0.4 * T)
    warmup = 20
    exposure = T - phase_one - warmup
    reference = closed[:phase_one].mean(axis=0)
    o1, o3, o4 = offsets(reference, sizes, basis)

    print(f"{T} simulated units, no batch structure | sizes "
          f"{sizes.min():.0f}-{sizes.max():.0f}, median {np.median(sizes):.0f}")
    print(f"the correct answer is no correlation\n")
    print(f"{'centring':<24}{'rate':>7}{'rho all':>10}{'p':>9}{'rho small':>11}{'p':>9}")
    rows = []
    test_sizes = sizes[phase_one:]
    small = test_sizes <= np.quantile(test_sizes, 0.25)
    for label, adjusted in (("single Phase I mean", coords),
                            ("first order", coords - o1),
                            ("first + incomplete second order", coords - o1 - o3),
                            ("first + complete second order", coords - o1 - o3 - o4)):
        fit = VC.calibrate(adjusted[:phase_one], closed[:phase_one], sizes[:phase_one],
                           target_far=NOMINAL / 1000.0, estimator="em", seed=0)
        out = VC.run(fit, adjusted[phase_one:], closed[phase_one:], sizes[phase_one:])
        stat = np.asarray(out["statistic"])
        rate = 1000.0 * int(out["alarm"][warmup:].sum()) / exposure
        rho, pv = stats.spearmanr(test_sizes, stat)
        rho_s, pv_s = stats.spearmanr(test_sizes[small], stat[small])
        rows.append({"centring": label, "rate": rate, "rho": float(rho), "p": float(pv),
                     "rho_small": float(rho_s), "p_small": float(pv_s)})
        print(f"{label:<24}{rate:>7.1f}{rho:>+10.3f}{pv:>9.4f}{rho_s:>+11.3f}{pv_s:>9.4f}")
    json.dump({"nominal_per_1000": NOMINAL, "T": T, "exposure": exposure,
               "batch_structure": False, "rows": rows},
              open(RESULTS / "centring_simulation.json", "w"), indent=2)
    print("\nwrote results/centring_simulation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
