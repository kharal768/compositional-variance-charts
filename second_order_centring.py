#!/usr/bin/env python3
"""Does a second-order centring term close the sign flip at small units?

The first-order correction subtracts the O(1/K) term of E[log p-hat]. On real
lots it takes the rank correlation between unit size and the charting statistic
from +0.53 to a residual of the opposite sign at the smallest lots, which the
manuscript currently handles by excluding units below about five items. An
overshoot is what an omitted next term looks like, so the next term is worth
trying before an exclusion rule.

Expanding log of a multinomial proportion about p,

    E[log p-hat] = log p - (1-p)/(2Kp) + (1-p)(1-2p)/(3K^2 p^2) + O(K^-3),

the second term is the correction in use and the third is added here. Its sign
depends on whether p is below one half, so it can either deepen or reverse the
residual - which is the question.

Reported for each centring: the realised in-control rate and the Spearman
correlation between unit size and the statistic, over all units and over the
small units where the overshoot appears.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon import composition, data, varcomp as VC
from wmmon.composition import ilr_basis
from wmmon.paths import WMMON_HOME

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
NOMINAL = 5.0


def first_order(reference, sizes, basis):
    c = (1.0 - reference) / (2.0 * np.clip(reference, 1e-12, None))
    return -np.outer(1.0 / np.asarray(sizes, float), c) @ basis.T


def second_order(reference, sizes, basis):
    p = np.clip(reference, 1e-12, None)
    c2 = (1.0 - p) * (1.0 - 2.0 * p) / (3.0 * p ** 2)
    return np.outer(1.0 / np.asarray(sizes, float) ** 2, c2) @ basis.T


def main() -> int:
    seed = 0
    df, _ = data.load_lswmd(f"{WMMON_HOME}/data/subset.pkl")
    _, _, stream = data.split_labelled(df, seed=seed)
    stream = stream.reset_index(drop=True)
    sequence = data.permuted_lot_order(stream, seed=seed)[:1200]
    keep = set(sequence)
    lots_all = stream["lot"].to_numpy(dtype=object)
    idx = np.flatnonzero(np.array([str(l) in keep for l in lots_all]))
    lots = np.array([str(l) for l in lots_all[idx]], dtype=object)
    proba = np.load(CACHE / "cnnproba_clean_1200_perm.npz")["p"]

    frame = composition.build_stream(lots, proba, sequence, mode="hard")
    coords = composition.ilr_matrix(frame)
    props = composition.proportion_matrix(frame)
    sizes = frame["lot_size"].to_numpy().astype(float)
    basis = ilr_basis(props.shape[1])
    n_units = len(coords)
    phase_one = max(20 * coords.shape[1], int(0.4 * n_units))
    warmup = 20
    exposure = n_units - phase_one - warmup
    reference = props[:phase_one].mean(axis=0)

    o1 = first_order(reference, sizes, basis)
    o2 = second_order(reference, sizes, basis)
    variants = (("single Phase I mean", coords),
                ("first order", coords - o1),
                ("first + second order", coords - o1 - o2))

    print(f"{n_units} real lots | sizes {sizes.min():.0f}-{sizes.max():.0f}, "
          f"median {np.median(sizes):.0f} | exposure {exposure}\n")
    print(f"{'centring':<24}{'rate':>7}{'rho all':>10}{'p':>9}"
          f"{'rho small':>11}{'p':>9}")
    rows = []
    test_sizes = sizes[phase_one:]
    small = test_sizes <= np.quantile(test_sizes, 0.25)
    for label, adjusted in variants:
        fit = VC.calibrate(adjusted[:phase_one], props[:phase_one], sizes[:phase_one],
                           target_far=NOMINAL / 1000.0, estimator="em", seed=seed)
        out = VC.run(fit, adjusted[phase_one:], props[phase_one:], sizes[phase_one:])
        stat = np.asarray(out["statistic"])
        rate = 1000.0 * int(out["alarm"][warmup:].sum()) / exposure
        rho, pv = stats.spearmanr(test_sizes, stat)
        rho_s, pv_s = stats.spearmanr(test_sizes[small], stat[small])
        rows.append({"centring": label, "rate": rate, "rho": float(rho),
                     "p": float(pv), "rho_small": float(rho_s), "p_small": float(pv_s),
                     "n_small": int(small.sum())})
        print(f"{label:<24}{rate:>7.1f}{rho:>+10.3f}{pv:>9.4f}"
              f"{rho_s:>+11.3f}{pv_s:>9.4f}")
    RESULTS.mkdir(exist_ok=True)
    json.dump({"nominal_per_1000": NOMINAL, "n_units": n_units,
               "exposure": exposure, "rows": rows},
              open(RESULTS / "second_order_centring.json", "w"), indent=2)
    print(f"\nsmall units = smallest quartile by size ({int(small.sum())} units)")
    print("wrote results/second_order_centring.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
