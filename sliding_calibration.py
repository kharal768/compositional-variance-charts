#!/usr/bin/env python3
"""Is WM-811K's residual alarm rate estimator failure or real drift?

An earlier configuration left the corrected chart at 212 per 1,000 on WM-811K against a
nominal 5, with two candidate explanations that the study could not separate:
the dispersion estimator understates over-dispersion when units carry lot
structure, or the stream contains real drift that no in-control calibration
should absorb.

They make opposite predictions about LOCAL recalibration. If the excess is slow
drift, a chart recalibrated on the immediately preceding window is always
comparing like with like and the rate should fall toward nominal as the window
shortens. If the estimator is failing, the miscalibration is present in every
window regardless of length and the rate should stay high.

So the calibration is slid along the stream: before each block of units, the
chart is refitted on the preceding ``window`` units only, and alarms are
counted on that block. No degradation is induced anywhere. A short window
costs precision, so a floor is imposed at 20 units per monitored dimension ---
below that the fit is too noisy to interpret.

Run on all three datasets: a mechanism that only shows up on WM-811K is a
property of that stream, and one that shows up everywhere is a property of the
method.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

import phase1_ratio_sweep as P
from wmmon import composition, laney_coda as L

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
NOMINAL = 5.0


def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo * 1000, hi * 1000


def stream_coords(proba, unit):
    n_units = len(proba) // unit
    proba = proba[: n_units * unit]
    ids = np.array([f"u{i // unit:06d}" for i in range(len(proba))], dtype=object)
    seq = [f"u{i:06d}" for i in range(n_units)]
    frame = composition.build_stream(ids, proba, seq, mode="soft")
    return (composition.ilr_matrix(frame),
            composition.proportion_matrix(frame),
            frame["lot_size"].to_numpy())


def sliding(coords, props, sizes, window, block=25, seed=0, lag=1):
    """Refit on the preceding `window` units, then score the next `block`.

    ``lag`` is passed to the dispersion estimator. The two corrections address
    different components --- the window handles slow non-stationarity, the lag
    handles clustering within the calibration window --- so they are expected
    to compose rather than substitute.
    """
    dim = coords.shape[1]
    if window < 20 * dim:
        return None
    alarms = exposure = 0
    sigmas = []
    start = window
    while start + block <= len(coords):
        fit = L.calibrate(coords[start - window:start], props[start - window:start],
                          sizes[start - window:start],
                          target_far=NOMINAL / 1000.0, lag=lag, seed=seed)
        out = L.run(fit, coords[start:start + block], props[start:start + block],
                    sizes[start:start + block])
        alarms += int(out["alarm"].sum())
        exposure += block
        sigmas.append(fit.inflation)
        start += block
    if exposure < 90:
        return None
    lo, hi = clopper_pearson(alarms, exposure)
    return {"window": window, "lag": lag, "alarms": alarms, "exposure": exposure,
            "rate": 1000.0 * alarms / exposure, "ci": [lo, hi],
            "covers": bool(lo <= NOMINAL <= hi),
            "mean_sigma_Z": float(np.mean(sigmas))}


def main() -> None:
    t0 = time.time()
    seed = 0
    results = []

    configs = [("wm811k", 24), ("digits", 3), ("image_segments", 4)]
    for name, unit in configs:
        proba = (np.load(CACHE / "cnnproba_clean_1200_perm.npz")["p"]
                 if name == "wm811k" else P.probabilities(name, seed))
        coords, props, sizes = stream_coords(proba, unit)
        dim = coords.shape[1]
        print(f"\n{name} (unit {unit}, {len(coords)} units, dim {dim})", flush=True)
        windows = [w for w in (20 * dim, 40 * dim, 80 * dim, 160 * dim)
                   if w + 100 < len(coords)]
        for w in windows:
            for lag in (1, 4):
                row = sliding(coords, props, sizes, w, seed=seed, lag=lag)
                if not row:
                    continue
                row.update(dataset=name, unit=unit, dim=dim,
                           units_per_dim=round(w / dim))
                results.append(row)
                print(f"  window {w:>4} ({row['units_per_dim']:>3}/dim) lag {lag} "
                      f"rate {row['rate']:7.1f} "
                      f"CI [{row['ci'][0]:.1f}, {row['ci'][1]:.1f}] "
                      f"{'covers' if row['covers'] else 'MISS'} "
                      f"| mean sigma_Z {row['mean_sigma_Z']:.2f}", flush=True)

    # No automatic verdict is printed: a 25%
    # reduction as "drift", which reads as an explanation when the residual
    # was still 44x nominal. The numbers above are reported without a verdict.
    print("\nbest cell per dataset (lowest realised rate):")
    for name, _ in configs:
        rows = [r for r in results if r["dataset"] == name]
        if not rows:
            continue
        best = min(rows, key=lambda r: r["rate"])
        print(f"  {name:<16s} window {best['window']:>4} lag {best['lag']} -> "
              f"{best['rate']:7.1f} per 1,000 "
              f"{'(covers nominal)' if best['covers'] else '(misses nominal)'}")

    RESULTS.mkdir(exist_ok=True)
    json.dump({"nominal_per_1000": NOMINAL, "rows": results},
              open(RESULTS / "sliding_calibration.json", "w"), indent=2)
    print(f"\nwrote results/sliding_calibration.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
