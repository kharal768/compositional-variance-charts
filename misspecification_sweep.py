#!/usr/bin/env python3
"""Does misspecifying the known sampling term move the false-alarm rate?

The paper's central methodological claim is that error in the closed-form
within-unit covariance is absorbed into the estimated between-unit term rather
than passed to the false-alarm rate. Until now that rested on one observation:
the closed form is about 41% out at 24 items per unit and the chart still holds
its nominal rate.

This scales the sampling covariance by a factor k before calibration and
scoring, so the term the chart treats as known is deliberately wrong by a stated
amount, and records the realised in-control rate at each k. Yashchin (1995,
Table 2) reports the comparison case for a scheme that assumes its within
variance: at k = 2 the in-control ARL falls from 250 to 37.9, about 6.6 times
the intended false-alarm rate.

Both estimators are run. The moment estimator clips negative eigenvalues, so it
cannot absorb an over-large known term; maximum likelihood estimates the between
term jointly and can.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon import composition, varcomp as VC
from wmmon.paths import WMMON_HOME

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
NOMINAL = 5.0
UNIT = 24
FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)


def interval(k, n):
    lo = stats.beta.ppf(0.025, k, n - k + 1) * 1000 if k else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) * 1000 if k < n else 1000.0
    return lo, hi


def run(scale, estimator, coords, props, sizes, phase_one, warmup):
    """Calibrate and score with the sampling covariance multiplied by `scale`."""
    fit = VC.calibrate(coords[:phase_one], props[:phase_one], sizes[:phase_one],
                       target_far=NOMINAL / 1000.0, estimator=estimator,
                       lag=4, seed=0, sampling_scale=scale)
    out = VC.run(fit, coords[phase_one:], props[phase_one:], sizes[phase_one:],
                 sampling_scale=scale)
    alarms = int(out["alarm"][warmup:].sum())
    exposure = len(coords) - phase_one - warmup
    lo, hi = interval(alarms, exposure)
    return {"scale": scale, "estimator": estimator,
            "rate": 1000.0 * alarms / exposure, "ci": [lo, hi],
            "covers": bool(lo <= NOMINAL <= hi),
            "between_fraction": float(fit.between_fraction)}


def main() -> int:
    proba = np.load(CACHE / "cnnproba_clean_1200_perm.npz")["p"]
    n_units = len(proba) // UNIT
    p = proba[: n_units * UNIT]
    ids = np.array([f"u{i // UNIT:06d}" for i in range(len(p))], dtype=object)
    seq = [f"u{i:06d}" for i in range(n_units)]
    frame = composition.build_stream(ids, p, seq, mode="hard")
    coords = composition.ilr_matrix(frame)
    props = composition.proportion_matrix(frame)
    sizes = frame["lot_size"].to_numpy()
    phase_one, warmup = max(160, int(0.4 * n_units)), 20

    rows = []
    print(f"WM-811K, {UNIT} wafers per unit, nominal {NOMINAL:.1f} per 1,000\n")
    print(f"{'scale k':>8}{'estimator':>12}{'rate':>9}{'95% CI':>18}"
          f"{'covers':>9}{'batch share':>13}")
    for estimator in ("em", "moment"):
        for scale in FACTORS:
            r = run(scale, estimator, coords, props, sizes, phase_one, warmup)
            rows.append(r)
            ci = f"[{r['ci'][0]:.1f}, {r['ci'][1]:.1f}]"
            print(f"{scale:>8.2f}{estimator:>12}{r['rate']:>9.1f}{ci:>18}"
                  f"{'yes' if r['covers'] else 'NO':>9}"
                  f"{100 * r['between_fraction']:>12.1f}%")
    RESULTS.mkdir(exist_ok=True)
    json.dump({"nominal_per_1000": NOMINAL, "unit": UNIT, "rows": rows},
              open(RESULTS / "misspecification_sweep.json", "w"), indent=2)
    print("\nwrote results/misspecification_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
