#!/usr/bin/env python3
"""How much between-unit signal is needed before the components separate?

The method section states a rule - do not interpret component-wise
estimates below a trace ratio of roughly 0.4 - that rested on two simulation
points. Two points cannot establish a threshold. This sweeps seven unit sizes,
holding the true between-unit covariance fixed and letting the sampling
covariance shrink with n, and reports two errors separately:

  total      how well the estimator recovers the OVERALL between-unit variance,
             which is what the control limit depends on;
  component  how well it recovers the individual diagonal entries, which is
             what a variance decomposition reported to a reader depends on.

The distinction matters because the realised false-alarm rate was found
unaffected by estimator choice while the decomposition differed: the chart uses
the sum of the two covariances, the decomposition uses the split.
"""
from wmmon.paths import WMMON_HOME
import json
import numpy as np
from wmmon import varcomp as VC
from wmmon.composition import ilr_basis
from wmmon.laney_coda import sampling_covariance

V = ilr_basis(5)
P = np.array([.5, .2, .15, .1, .05])
T = 1500
TRUE = np.diag([0.02, 0.01, 0.005, 0.002])

header = f"{'n/unit':>7}{'trace ratio':>13}{'total err':>11}{'component err':>15}"
print(header + "  estimated diagonal")
rows = []
for n_unit in (25, 50, 100, 200, 400, 800):
    rng = np.random.default_rng(0)
    samp = np.array([sampling_covariance(P, float(n_unit), V) for _ in range(T)])
    ratio = float(np.trace(TRUE) / np.trace(samp[0]))
    b = rng.multivariate_normal(np.zeros(4), TRUE, size=T)
    e = np.array([rng.multivariate_normal(np.zeros(4), samp[t]) for t in range(T)])
    est, _, _ = VC.fit_between_em(b + e, np.zeros(4), samp, iterations=600)
    d, td = np.diag(est), np.diag(TRUE)
    total = float(abs(d.sum() - td.sum()) / td.sum())
    comp = float(np.mean(np.abs(d - td) / td))
    rows.append({"n_unit": n_unit, "trace_ratio": ratio, "total_rel_error": total,
                 "component_rel_error": comp, "estimated": [float(x) for x in d]})
    print(f"{n_unit:>7}{ratio:>13.3f}{total:>11.3f}{comp:>15.3f}  {np.round(d, 4)}")

print(f"\ntruth {np.round(np.diag(TRUE), 4)}")
json.dump({"truth": [float(x) for x in np.diag(TRUE)], "rows": rows},
          open(f"{WMMON_HOME}/results/identifiability_sweep.json", "w"), indent=2)
print("wrote results/identifiability_sweep.json")
