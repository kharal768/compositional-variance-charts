#!/usr/bin/env python3
"""Does the batch-structure mechanism hold under count aggregation?

The evidence for the paper's unification claim - that three separately reported
calibration failures share one mechanism - was produced with probability-
aggregated compositions, which Finding 1 shows is a misspecified configuration.
This re-runs the decisive test under counts: if permuting the wafer stream at
item level (destroying batch structure, keeping every marginal) brings the
uncorrected chart to its nominal rate while the lot-structured stream fails,
batch structure is the mechanism under the correct configuration too.
"""
from wmmon.paths import WMMON_HOME
import json
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon import composition, laney_coda as L

CACHE = Path(f"{WMMON_HOME}/cache"); RESULTS = Path(f"{WMMON_HOME}/results")
NOM = 5.0


def cpi(k, n):
    lo = stats.beta.ppf(0.025, k, n - k + 1) * 1000 if k else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) * 1000 if k < n else 1000.0
    return lo, hi


def run(proba, unit, mode):
    n = len(proba) // unit
    p = proba[: n * unit]
    ids = np.array([f"u{i // unit:06d}" for i in range(len(p))], dtype=object)
    seq = [f"u{i:06d}" for i in range(n)]
    fr = composition.build_stream(ids, p, seq, mode=mode)
    C = composition.ilr_matrix(fr); P = composition.proportion_matrix(fr)
    S = fr["lot_size"].to_numpy()
    ph = max(20 * C.shape[1], int(0.4 * n)); ex = n - ph - 20
    fit = L.calibrate(C[:ph], P[:ph], S[:ph], target_far=NOM / 1000, seed=0)
    k = int(L.run_uncorrected(fit, C[ph:], P[ph:], S[ph:])["alarm"][20:].sum())
    lo, hi = cpi(k, ex)
    return {"rate": 1000 * k / ex, "ci": [lo, hi], "exposure": ex,
            "covers": bool(lo <= NOM <= hi)}


proba = np.load(CACHE / "cnnproba_clean_1200_perm.npz")["p"]
shuffled = proba[np.random.default_rng(0).permutation(len(proba))]
rows = []
print(f"{'aggregation':<12}{'unit':>5}  {'lot-structured':>30}  {'item-shuffled':>30}")
for mode in ("hard", "soft"):
    for unit in (24, 48):
        a = run(proba, unit, mode); b = run(shuffled, unit, mode)
        rows.append({"mode": mode, "unit": unit, "lot_structured": a, "item_shuffled": b})
        fa = f"{a['rate']:6.1f} [{a['ci'][0]:.1f},{a['ci'][1]:.1f}] {'ok' if a['covers'] else 'MISS'}"
        fb = f"{b['rate']:6.1f} [{b['ci'][0]:.1f},{b['ci'][1]:.1f}] {'ok' if b['covers'] else 'MISS'}"
        print(f"{mode:<12}{unit:>5}  {fa:>30}  {fb:>30}")
json.dump({"nominal_per_1000": NOM, "chart": "uncorrected multinomial", "rows": rows},
          open(RESULTS / "mechanism_counts.json", "w"), indent=2)
print("\nwrote results/mechanism_counts.json")
