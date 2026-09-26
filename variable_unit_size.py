#!/usr/bin/env python3
"""Does the chart survive variable monitoring-unit sizes?

Every calibration experiment in this study used fixed-size units --- blocks of
24 or 48 wafers --- so the expected value of the ILR coordinates was the same
for every unit and the Phase I empirical mean absorbed it. Production lots are
not fixed size.

Kartiosuo et al. (2025) give the size dependence explicitly. For a composition
from K counts, the expectation of the log-proportions is approximately

    E(log p_j) ~= log(a_j) - (1 / (2 K)) * (1 - a_j) / a_j

in the multinomial limit, so the expected ILR vector is

    E(ilr) ~= V (log a - c / K),     c_j = (1 - a_j) / (2 a_j),

which shifts with K. When units differ in size, no single centre is correct for
all of them: small units are biased away from the Phase I mean in a direction
that depends on the composition, and the chart sees that bias as a signal.

Two things are measured here on WM-811K, using **real production lots** as the
monitoring units so that the size distribution is the one the data actually has:

1. the realised false-alarm rate with a single Phase I mean, as the method
   currently stands;
2. the same with a per-unit mean correction applied, which is the published
   approximation used as intended.

If the correction matters, it belongs in the method; if it does not, the
fixed-mean simplification should be stated as verified rather than assumed.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import json
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon import composition, data, varcomp as VC
from wmmon.composition import ilr_basis

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
NOMINAL = 5.0


def clopper_pearson(k, n):
    lo = stats.beta.ppf(0.025, k, n - k + 1) * 1000 if k else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) * 1000 if k < n else 1000.0
    return lo, hi


def mean_offset(reference: np.ndarray, sizes: np.ndarray,
                basis: np.ndarray) -> np.ndarray:
    """Size-dependent offset of the expected ILR vector, one row per unit."""
    c = (1.0 - reference) / (2.0 * np.clip(reference, 1e-12, None))
    return -np.outer(1.0 / np.asarray(sizes, dtype=float), c) @ basis.T


def main() -> None:
    seed = 0
    df, _ = data.load_lswmd(f"{WMMON_HOME}/data/subset.pkl")
    _, _, stream = data.split_labelled(df, seed=seed)
    stream = stream.reset_index(drop=True)

    sequence = data.permuted_lot_order(stream, seed=seed)[:1200]
    keep = set(sequence)
    lots_all = stream["lot"].to_numpy(dtype=object)
    rows_idx = np.flatnonzero(np.array([str(l) in keep for l in lots_all]))
    lots = np.array([str(l) for l in lots_all[rows_idx]], dtype=object)
    proba = np.load(CACHE / "cnnproba_clean_1200_perm.npz")["p"]

    # Real lots as monitoring units: the size distribution is the data's own.
    frame = composition.build_stream(lots, proba, sequence, mode="hard")
    coords = composition.ilr_matrix(frame)
    props = composition.proportion_matrix(frame)
    sizes = frame["lot_size"].to_numpy().astype(float)
    basis = ilr_basis(props.shape[1])
    n_units = len(coords)
    phase_one = max(20 * coords.shape[1], int(0.4 * n_units))
    warmup = 20
    exposure = n_units - phase_one - warmup

    print(f"{n_units} real lots as units | size: median {np.median(sizes):.0f}, "
          f"range {sizes.min():.0f}-{sizes.max():.0f}, "
          f"IQR {np.percentile(sizes, 25):.0f}-{np.percentile(sizes, 75):.0f}")
    print(f"Phase I {phase_one}, exposure {exposure}\n")

    reference = props[:phase_one].mean(axis=0)
    offset = mean_offset(reference, sizes, basis)
    spread = float(np.linalg.norm(offset[sizes.argmin()] - offset[sizes.argmax()]))
    print(f"predicted centre shift between the smallest and largest unit: "
          f"{spread:.4f} in ILR distance")

    results = []
    for label, adjusted in (("single Phase I mean", coords),
                            ("per-unit mean correction", coords - offset)):
        fit = VC.calibrate(adjusted[:phase_one], props[:phase_one],
                           sizes[:phase_one], target_far=NOMINAL / 1000.0,
                           estimator="em", seed=seed)
        alarm = VC.run(fit, adjusted[phase_one:], props[phase_one:],
                       sizes[phase_one:])["alarm"]
        k = int(alarm[warmup:].sum())
        lo, hi = clopper_pearson(k, exposure)
        results.append({"centring": label, "alarms": k, "exposure": exposure,
                        "rate": 1000.0 * k / exposure, "ci": [lo, hi],
                        "covers": bool(lo <= NOMINAL <= hi)})
        print(f"   {label:<26s} rate {1000.0 * k / exposure:6.1f} "
              f"CI [{lo:.1f}, {hi:.1f}] "
              f"{'covers' if results[-1]['covers'] else 'MISS'}")

    # Does the chart's statistic correlate with unit size? If the centring is
    # wrong, small units should look anomalous simply for being small.
    for label, adjusted in (("single Phase I mean", coords),
                            ("per-unit mean correction", coords - offset)):
        fit = VC.calibrate(adjusted[:phase_one], props[:phase_one],
                           sizes[:phase_one], target_far=NOMINAL / 1000.0,
                           estimator="em", seed=seed)
        stat = VC.run(fit, adjusted[phase_one:], props[phase_one:],
                      sizes[phase_one:])["statistic"]
        rho, p = stats.spearmanr(sizes[phase_one:], stat)
        print(f"   {label:<26s} Spearman(unit size, statistic) = "
              f"{rho:+.3f}, p = {p:.4f}")
        results[[r["centring"] for r in results].index(label)].update(
            spearman_rho=float(rho), spearman_p=float(p))

    RESULTS.mkdir(exist_ok=True)
    json.dump({"nominal_per_1000": NOMINAL, "n_units": n_units,
               "size_median": float(np.median(sizes)),
               "size_min": float(sizes.min()), "size_max": float(sizes.max()),
               "centre_shift": spread, "rows": results},
              open(RESULTS / "variable_unit_size.json", "w"), indent=2)
    print("\nwrote results/variable_unit_size.json")


if __name__ == "__main__":
    main()
