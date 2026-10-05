#!/usr/bin/env python3
"""Why does the bootstrap limit miss nominal, and is the miss predictable?

The empirical bootstrap limit was found to realise anything from 7 to
180 false alarms per 1,000 against a nominal 5, depending on configuration, and
isolated the cause to Phase I length and unit construction rather than the
monitored model. That is a defect if left there, and a result if characterised.

Hypothesis: the limit is estimated from a Phase I covariance in (D-1)
dimensions. When Phase I is short relative to that dimension the covariance is
underestimated, the limit comes out too low, and the chart is anticonservative.
The governing quantity should then be the RATIO of Phase I units to monitored
dimension, not the absolute number of units --- which would explain why the
same rule behaves differently on datasets with different class counts.

Protocol, fixed before looking at any output:

  EXPLORE on `digits` (9 dimensions) and `image_segments` (6). Sweep the ratio,
  and if realised rate converges to nominal above some ratio, read that
  threshold off the exploration datasets only.

  CONFIRM on WM-811K (8 dimensions), which is not consulted while the threshold
  is chosen. A rule read off the exploration set and then tested once on the
  held-out set is a prediction; a rule fitted to all three is a description.

Streams are shuffled and undegraded throughout, so every alarm is a false alarm.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from wmmon import composition, monitor

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
NOMINAL = 5.0
RATIOS = [5, 10, 20, 40, 80]


def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo * 1000, hi * 1000


def tabular(name):
    if name == "digits":
        from sklearn.datasets import load_digits
        d = load_digits()
        return d.data.astype(float), d.target.astype(int)
    import itertools
    from river import datasets
    rows = list(itertools.islice(iter(datasets.ImageSegments()), 10**6))
    keys = sorted(rows[0][0])
    x = np.array([[r[0][k] for k in keys] for r in rows], dtype=float)
    labels = sorted({r[1] for r in rows})
    idx = {l: i for i, l in enumerate(labels)}
    return x, np.array([idx[r[1]] for r in rows])


def probabilities(name, seed=0):
    if name == "wm811k":
        return np.load(CACHE / "cnnproba_clean_1200_perm.npz")["p"]
    x, y = tabular(name)
    classes = sorted(np.unique(y))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    n_train = int(0.3 * len(x))
    scaler = StandardScaler().fit(x[:n_train])
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    clf.fit(scaler.transform(x[:n_train]), y[:n_train],
            sample_weight=compute_sample_weight("balanced", y[:n_train]))
    p = clf.predict_proba(scaler.transform(x[n_train:]))
    out = np.zeros((len(x) - n_train, len(classes)))
    for j, c in enumerate(clf.classes_):
        out[:, int(c)] = p[:, j]
    return out


def sweep(name, unit, seed=0):
    proba = probabilities(name, seed)
    n_units = len(proba) // unit
    proba = proba[: n_units * unit]
    ids = np.array([f"u{i // unit:06d}" for i in range(len(proba))], dtype=object)
    seq = [f"u{i:06d}" for i in range(n_units)]
    frame = composition.build_stream(ids, proba, seq, mode="soft")
    coords = composition.ilr_matrix(frame)
    sizes = frame["lot_size"].to_numpy()
    dim = coords.shape[1]
    warmup = 20

    rows = []
    for ratio in RATIOS:
        phase_one = ratio * dim
        exposure = n_units - phase_one - warmup
        if phase_one < 40 or exposure < 90:
            continue
        fit = monitor.calibrate(coords[:phase_one], sizes[:phase_one], lam=0.2,
                                target_far=NOMINAL / 1000.0, seed=seed)
        alarm = monitor.run_monitor(
            fit, coords[phase_one:], sizes[phase_one:])["alarm"]
        k = int(alarm[warmup:].sum())
        lo, hi = clopper_pearson(k, exposure)
        rows.append({"dataset": name, "unit": unit, "dim": dim, "ratio": ratio,
                     "phase_one": phase_one, "exposure": exposure, "alarms": k,
                     "rate": 1000.0 * k / exposure, "ci": [lo, hi],
                     "covers": bool(lo <= NOMINAL <= hi)})
    return rows


def main() -> None:
    t0 = time.time()
    explore, confirm = [], []

    print("EXPLORATION (digits, image_segments)")
    for name, unit in (("digits", 3), ("image_segments", 4)):
        for r in sweep(name, unit):
            explore.append(r)
            print(f"  {name:<15s} dim {r['dim']} ratio {r['ratio']:>3} "
                  f"(Phase I {r['phase_one']:>4}) exposure {r['exposure']:>4} "
                  f"rate {r['rate']:>7.1f} CI [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}] "
                  f"{'covers' if r['covers'] else 'MISS'}", flush=True)

    # Threshold read off the exploration set only.
    by_ratio = {}
    for r in explore:
        by_ratio.setdefault(r["ratio"], []).append(r["covers"])
    fully = [ratio for ratio, flags in sorted(by_ratio.items()) if all(flags)]
    threshold = fully[0] if fully else None
    print(f"\n  ratios covering nominal in every exploration cell: {fully}")
    print(f"  chosen threshold: {threshold}")

    print("\nCONFIRMATION (wm811k, held out)")
    for unit in (24, 48):
        for r in sweep("wm811k", unit):
            confirm.append(r)
            print(f"  wm811k unit {r['unit']:>2} dim {r['dim']} ratio {r['ratio']:>3} "
                  f"(Phase I {r['phase_one']:>4}) exposure {r['exposure']:>4} "
                  f"rate {r['rate']:>7.1f} CI [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}] "
                  f"{'covers' if r['covers'] else 'MISS'}", flush=True)

    verdict = "not evaluated"
    if threshold is not None:
        tested = [r for r in confirm if r["ratio"] >= threshold]
        if tested:
            passed = sum(r["covers"] for r in tested)
            verdict = f"{passed}/{len(tested)} held-out cells at ratio >= {threshold}"
            print(f"\n  PREDICTION: at Phase I >= {threshold} units per dimension "
                  f"the realised rate covers nominal.")
            print(f"  HELD-OUT RESULT: {verdict}")
        else:
            verdict = f"no held-out cell reaches ratio {threshold}"
            print(f"\n  {verdict}")

    RESULTS.mkdir(exist_ok=True)
    json.dump({"nominal_per_1000": NOMINAL, "threshold_ratio": threshold,
               "verdict": verdict, "exploration": explore, "confirmation": confirm},
              open(RESULTS / "phase1_ratio_sweep.json", "w"), indent=2)
    print(f"\nwrote results/phase1_ratio_sweep.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
