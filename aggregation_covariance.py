#!/usr/bin/env python3
"""Source for Finding 1: how far probability aggregation moves the reference covariance.

Summing predicted probabilities instead of counting arg-max predictions
inflates the multinomial reference covariance: 29-fold overall and 75-fold in
the rarest balance. This computes both ratios and writes them to a result
file.
"""
from wmmon.paths import WMMON_HOME
import json
from pathlib import Path

import numpy as np

from wmmon import composition
from wmmon.composition import ilr_basis
from wmmon.laney_coda import sampling_covariance

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
UNIT = 24

proba = np.load(CACHE / "cnnproba_clean_1200_perm.npz")["p"]
n = len(proba) // UNIT
p = proba[: n * UNIT]
ids = np.array([f"u{i // UNIT:06d}" for i in range(len(p))], dtype=object)
seq = [f"u{i:06d}" for i in range(n)]
phase_one = max(160, int(0.4 * n))

out = {}
for mode in ("soft", "hard"):
    frame = composition.build_stream(ids, p, seq, mode=mode)
    props = composition.proportion_matrix(frame)
    ref = props[:phase_one].mean(axis=0)
    V = ilr_basis(props.shape[1])
    cov = sampling_covariance(ref, float(UNIT), V)
    out[mode] = {"min_mean_part": float(ref.min()),
                 "trace_sampling_covariance": float(np.trace(cov)),
                 "max_diagonal": float(np.diag(cov).max())}
ratio = out["soft"]["trace_sampling_covariance"] / out["hard"]["trace_sampling_covariance"]
diag_ratio = out["soft"]["max_diagonal"] / out["hard"]["max_diagonal"]
out["trace_ratio_soft_over_hard"] = ratio
out["max_diagonal_ratio_soft_over_hard"] = diag_ratio
for mode in ("soft", "hard"):
    r = out[mode]
    print(f"{mode:<5s} min mean part {r['min_mean_part']:.4f} | trace {r['trace_sampling_covariance']:8.2f} | max diagonal {r['max_diagonal']:.3f}")
print(f"trace ratio {ratio:.1f}x | largest-diagonal ratio {diag_ratio:.1f}x")
json.dump(out, open(RESULTS / "aggregation_covariance.json", "w"), indent=2)
print("wrote results/aggregation_covariance.json")
