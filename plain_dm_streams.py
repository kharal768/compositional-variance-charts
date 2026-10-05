#!/usr/bin/env python3
"""The plain individuals T^2 with the F limit, on the streams of the dispersion validation.

estimator_comparison.py tests the dispersion estimators on five-part streams of 25 items per unit, either
pure multinomial, Dirichlet-multinomial at several concentrations, or a two-regime mixture that no
Dirichlet can represent. These are non-Gaussian forms of over-dispersion, unlike the Gaussian
between-unit variation used elsewhere, so they are a second test of the normality the F limit assumes.
The streams are identical to those of estimator_comparison.py (same generator, same seed), the first 800
units are Phase I, and the remaining 3,200 are evaluated.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from estimator_comparison import make_stream
from wmmon.paths import WMMON_HOME

NOMINAL, PHASE_ONE = 5.0, 800
cases = [("multinomial", None), ("dirichlet", 120.0), ("dirichlet", 60.0), ("dirichlet", 30.0),
         ("dirichlet", 20.0), ("dirichlet", 12.0), ("dirichlet", 8.0),
         ("mixture", 0.05), ("mixture", 0.10), ("mixture", 0.15)]
STREAMS = 60
rows = []
print(f"{STREAMS} independent streams per case; Phase I {PHASE_ONE}, 3,200 evaluated units each\n")
print(f"{'stream':<20}{'F limit rate':>13}{'95% CI':>18}{'q10 / median / q90 across streams':>36}{'empirical':>11}")
for kind, param in cases:
    alarms = alarms_e = expo = 0
    per = []
    for s in range(STREAMS):
        counts, coords, closed, sizes = make_stream(kind, param, seed=s)
        D = coords.shape[1]
        mu = coords[:PHASE_ONE].mean(axis=0)
        inv = np.linalg.pinv(np.cov(coords[:PHASE_ONE], rowvar=False))
        d = coords - mu
        stat = np.einsum("ij,jk,ik->i", d, inv, d)
        m = PHASE_ONE
        f_lim = (D * (m + 1) * (m - 1) / (m * (m - D))) * stats.f.ppf(1 - NOMINAL / 1000, D, m - D)
        e_lim = float(np.quantile(stat[:m], 1 - NOMINAL / 1000))
        ev = stat[m:]
        k = int((ev > f_lim).sum())
        alarms += k; alarms_e += int((ev > e_lim).sum()); expo += len(ev)
        per.append(1000.0 * k / len(ev))
    lo = stats.beta.ppf(0.025, alarms, expo - alarms + 1) * 1000 if alarms else 0.0
    hi = stats.beta.ppf(0.975, alarms + 1, expo - alarms) * 1000
    per = np.array(per)
    rows.append({"stream": kind, "param": param, "streams": STREAMS, "alarms": alarms, "exposure": expo,
                 "rate_f": 1000.0 * alarms / expo, "ci_f": [lo, hi],
                 "q10": float(np.quantile(per, .1)), "median": float(np.quantile(per, .5)),
                 "q90": float(np.quantile(per, .9)), "rate_empirical": 1000.0 * alarms_e / expo})
    label = kind if param is None else f"{kind} {param:g}"
    print(f"{label:<20}{1000.0*alarms/expo:>13.2f}{f'[{lo:.2f}, {hi:.2f}]':>18}"
          f"{np.quantile(per,.1):>12.1f} /{np.quantile(per,.5):>6.1f} /{np.quantile(per,.9):>6.1f}{1000.0*alarms_e/expo:>11.2f}", flush=True)
out = Path(f"{WMMON_HOME}/results"); out.mkdir(exist_ok=True)
json.dump({"phase_one": PHASE_ONE, "streams": STREAMS, "rows": rows}, open(out / "plain_dm_streams.json", "w"), indent=2)
print("\nwrote results/plain_dm_streams.json")
