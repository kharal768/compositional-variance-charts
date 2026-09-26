#!/usr/bin/env python3
"""The calibration result, checked on every dataset available.

One claim holds throughout: on an in-control stream, the
asymptotic chi-square limit and MD3's theta*sigma rule both miss their nominal
false-alarm rate by one to two orders of magnitude, while an empirical
bootstrap limit holds it. That was measured on WM-811K alone --- which is
precisely how the imbalance "finding" came apart.

So it is measured here on all three datasets, at two monitoring-unit sizes
each, with exact binomial intervals. No drift is induced anywhere: every stream
is shuffled, so it is in control by construction and every alarm counted is a
false alarm. A rule that holds its nominal rate should produce an interval
covering it in every cell.

If the pattern appears on one dataset and not the others, this claim goes the
way of the last one and the paper has nothing left that generalises.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from wmmon import composition, data, md3, monitor

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
NOMINAL = 5.0
THETA = 2.0


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


def evaluate(proba, oof_indicator, unit, seed, label):
    """Realised in-control false-alarm rate for the three rules."""
    n_units = len(proba) // unit
    if n_units < 260:
        return None
    proba = proba[: n_units * unit]
    ids = np.array([f"u{i // unit:06d}" for i in range(len(proba))], dtype=object)
    seq = [f"u{i:06d}" for i in range(n_units)]
    frame = composition.build_stream(ids, proba, seq, mode="soft")
    coords = composition.ilr_matrix(frame)
    sizes = frame["lot_size"].to_numpy()
    phase_one = max(20 * coords.shape[1], int(0.4 * n_units))
    warmup = 20
    exposure = n_units - phase_one - warmup
    if exposure < 80:
        return None

    fit = monitor.calibrate(coords[:phase_one], sizes[:phase_one], lam=0.2,
                            target_far=NOMINAL / 1000.0, seed=seed)
    run = monitor.run_monitor(fit, coords[phase_one:], sizes[phase_one:])
    boot_k = int(run["alarm"][warmup:].sum())
    chi2 = monitor.chi_square_limit(coords.shape[1], NOMINAL / 1000.0)
    chi_k = int((run["statistic"][warmup:] > chi2).sum())

    rng = np.random.default_rng(seed)
    draws = [float(oof_indicator[rng.choice(len(oof_indicator), size=unit,
                                            replace=False)].mean())
             for _ in range(2000)]
    sigma = float(np.std(draws, ddof=1))
    density = md3.lot_margin_density(md3.margin_indicator(proba), ids, seq)
    md3_k = int((md3.md3_score(density, phase_one, lam=0.2)[warmup:]
                 > THETA * sigma).sum())

    out = {"dataset": label, "unit": unit, "n_units": n_units,
           "exposure": exposure, "dim": int(coords.shape[1])}
    for rule, k in (("chi2", chi_k), ("md3_theta2", md3_k), ("bootstrap", boot_k)):
        lo, hi = clopper_pearson(k, exposure)
        out[rule] = {"alarms": k, "rate": 1000.0 * k / exposure,
                     "ci": [lo, hi], "covers_nominal": bool(lo <= NOMINAL <= hi)}
    return out


def main() -> None:
    t0 = time.time()
    seed = 0
    results = []

    for name, units in (("digits", [3, 5]), ("image_segments", [4, 6])):
        x, y = tabular(name)
        classes = [str(c) for c in range(len(np.unique(y)))]
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(x))
        x, y = x[order], y[order]
        n_train = int(0.3 * len(x))
        x_tr, y_tr, x_st = x[:n_train], y[:n_train], x[n_train:]

        scaler = StandardScaler().fit(x_tr)
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        clf.fit(scaler.transform(x_tr), y_tr,
                sample_weight=compute_sample_weight("balanced", y_tr))
        p = clf.predict_proba(scaler.transform(x_st))
        proba = np.zeros((len(x_st), len(classes)))
        for j, c in enumerate(clf.classes_):
            proba[:, int(c)] = p[:, j]

        oof = np.zeros(len(x_tr))
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(x_tr):
            ens = md3.train_random_subspace(
                scaler.transform(x_tr[tr]), [str(v) for v in y_tr[tr]],
                classes, seed=seed)
            oof[te] = md3.margin_indicator(
                ens.probabilities(scaler.transform(x_tr[te])))

        for unit in units:
            row = evaluate(proba, oof, unit, seed, name)
            if row:
                results.append(row)
                print(f"  {name:<15s} unit {unit:>2} | chi2 {row['chi2']['rate']:7.1f} "
                      f"| md3 {row['md3_theta2']['rate']:7.1f} "
                      f"| boot {row['bootstrap']['rate']:6.1f}", flush=True)

    # WM-811K, from cached CNN probabilities on the shuffled stream
    df, _ = data.load_lswmd(f"{WMMON_HOME}/data/subset.pkl")
    train, _, stream = data.split_labelled(df, seed=seed)
    proba_w = np.load(CACHE / "cnnproba_clean_1200_perm.npz")["p"]
    x_feat_train = np.load(CACHE / "train.npz")["x"]
    y_feat_train = train["label"].tolist()
    oof_w = np.zeros(len(x_feat_train))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(x_feat_train):
        ens = md3.train_random_subspace(
            x_feat_train[tr], [y_feat_train[i] for i in tr],
            list(data.CLASSES), seed=seed)
        oof_w[te] = md3.margin_indicator(ens.probabilities(x_feat_train[te]))
    for unit in (24, 48):
        row = evaluate(proba_w, oof_w, unit, seed, "wm811k")
        if row:
            results.append(row)
            print(f"  {'wm811k':<15s} unit {unit:>2} | chi2 {row['chi2']['rate']:7.1f} "
                  f"| md3 {row['md3_theta2']['rate']:7.1f} "
                  f"| boot {row['bootstrap']['rate']:6.1f}", flush=True)

    print(f"\nin-control false alarms per 1,000 units, nominal {NOMINAL:.1f}, "
          f"exact 95% CI")
    print(f"{'dataset':<16s}{'unit':>5s}{'rule':>12s}{'rate':>9s}{'95% CI':>20s}"
          f"{'covers':>8s}")
    covered = {"chi2": 0, "md3_theta2": 0, "bootstrap": 0}
    for r in results:
        for rule in ("chi2", "md3_theta2", "bootstrap"):
            c = r[rule]
            covered[rule] += int(c["covers_nominal"])
            lo, hi = c["ci"]
            interval = f"[{lo:.1f}, {hi:.1f}]"
            print(f"{r['dataset']:<16s}{r['unit']:>5d}{rule:>12s}{c['rate']:>9.1f}"
                  f"{interval:>20s}{'yes' if c['covers_nominal'] else 'NO':>8s}")
    print(f"\ncells covering nominal, out of {len(results)}:")
    for rule, n in covered.items():
        print(f"   {rule:<12s} {n}/{len(results)}")

    RESULTS.mkdir(exist_ok=True)
    json.dump({"nominal_per_1000": NOMINAL, "theta": THETA, "rows": results,
               "covered": covered},
              open(RESULTS / "calibration_across_datasets.json", "w"), indent=2)
    print(f"\nwrote results/calibration_across_datasets.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
