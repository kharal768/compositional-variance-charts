#!/usr/bin/env python3
"""Why does the plain chart alarm less often on small units than on large ones?

With a pooled covariance, small units have larger sampling variance, so one expects them to alarm
more. The simulations show the opposite. One candidate is zero replacement: at small counts most units
contain empty cells, replacement pulls their log-ratios towards the centre, and the variance of the
coordinates then grows more slowly than 1/n. This measures, by unit size, the ratio of the empirical
trace of the coordinates' sampling covariance to the closed-form trace, and the mean of the plain
T^2 statistic and its exceedance of the F limit, in a stream with between-unit variation.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon.composition import bayesian_multiplicative_replacement, ilr, ilr_basis
from wmmon.laney_coda import sampling_covariance
from wmmon.paths import WMMON_HOME

P = np.array([0.808, 0.053, 0.028, 0.024, 0.020, 0.017, 0.017, 0.017, 0.017]); P = P / P.sum()
V = ilr_basis(len(P)); DIM = len(P) - 1
SIZES = (4, 8, 12, 18, 24, 32, 40)
DRAWS, NOMINAL, PHASE_ONE = 60000, 5.0, 240
rng = np.random.default_rng(5)

base = sampling_covariance(P, 25.0, V)
B = np.diag(np.linspace(1.0, 0.3, DIM)); B *= 0.30 * np.trace(base) / np.trace(B)
rows = []
# a pooled covariance and mean from a design with sizes 4-40, as in the plain-chart study
pool_sizes = rng.integers(4, 41, size=200000)
zs = []
for n in np.unique(pool_sizes):
    k = int((pool_sizes == n).sum())
    zs.append(ilr(bayesian_multiplicative_replacement(rng.multinomial(int(n), P, size=k).astype(float)))
              + rng.multivariate_normal(np.zeros(DIM), B, size=k))
Z = np.vstack(zs); mu = Z.mean(axis=0); inv = np.linalg.inv(np.cov(Z, rowvar=False))
f_lim = float(stats.chi2.ppf(1 - NOMINAL / 1000, DIM))   # the large-Phase-I limit of the F limit
print(f"{'n':>4}{'empty-cell share':>18}{'trace: empirical / closed form':>32}{'mean T2':>9}{'rate per 1,000':>16}")
for n in SIZES:
    counts = rng.multinomial(n, P, size=DRAWS).astype(float)
    empty = float((counts == 0).any(axis=1).mean())
    zr = ilr(bayesian_multiplicative_replacement(counts))
    emp_trace = float(np.trace(np.cov(zr, rowvar=False)))
    closed = float(np.trace(sampling_covariance(P, float(n), V)))
    zt = zr + rng.multivariate_normal(np.zeros(DIM), B, size=DRAWS)
    t2 = np.einsum("ij,jk,ik->i", zt - mu, inv, zt - mu)
    rows.append({"n": n, "empty_cell_share": empty, "trace_ratio": emp_trace / closed,
                 "mean_t2": float(t2.mean()), "rate": 1000.0 * float((t2 > f_lim).mean())})
    r = rows[-1]
    print(f"{n:>4}{100*empty:>17.0f}%{r['trace_ratio']:>32.3f}{r['mean_t2']:>9.2f}{r['rate']:>16.2f}")
Path(f"{WMMON_HOME}/results").mkdir(exist_ok=True)
json.dump({"rows": rows, "limit": f_lim}, open(Path(f"{WMMON_HOME}/results")/"size_class_mechanism.json", "w"), indent=2)
