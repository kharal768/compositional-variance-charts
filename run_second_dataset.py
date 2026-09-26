#!/usr/bin/env python3
"""Does any of this generalise beyond wafer maps?

Everything so far rests on one dataset. This runs the same protocol on two
others, neither of which is a wafer map:

``digits``          scikit-learn's bundled handwritten digits (1,797 samples,
                    64 features, 10 classes). Sethi & Kantardzic used Digits08
                    and Digits17 from the same UCI family, so results here sit
                    beside theirs.
``image_segments``  river's bundled image-segmentation benchmark (2,310
                    samples, 18 features, 7 classes).

Both are small. That is a real limitation and is reported rather than hidden:
they are what is reachable from this environment, and they establish that the
protocol and the calibration finding are not artefacts of WM-811K, not that the
method works at scale on arbitrary data.

Drift is induced by the Sethi protocol --- rotate the top 25% of features by
information gain (a real drift, must be detected) or the bottom 25% (an
irrelevant change, must be ignored). Using someone else's induction scheme on
someone else's data is the strongest generalisation evidence available here.

Monitoring units are formed from a shuffled sample order, so the pre-change
stream is in control by construction and a false alarm is genuinely a false
alarm --- the correction that changed several conclusions on WM-811K.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import matched_comparison as mc
from wmmon import composition, md3, monitor, sethi
from wmmon.monitor import _mewma_statistics, _shrunk_covariance

RESULTS = Path(f"{WMMON_HOME}/results")
TARGETS = [1.0, 5.0, 20.0]


def load(name: str):
    if name == "digits":
        from sklearn.datasets import load_digits

        d = load_digits()
        return d.data.astype(float), d.target.astype(int)
    if name == "image_segments":
        import itertools

        from river import datasets

        rows = list(itertools.islice(iter(datasets.ImageSegments()), 10**6))
        keys = sorted(rows[0][0])
        x = np.array([[r[0][k] for k in keys] for r in rows], dtype=float)
        labels = sorted({r[1] for r in rows})
        index = {l: i for i, l in enumerate(labels)}
        return x, np.array([index[r[1]] for r in rows])
    raise ValueError(name)


def proportion_statistic(props, sizes, phase_one):
    mean = props[:phase_one].mean(axis=0)
    scale = np.sqrt(sizes / np.median(sizes[:phase_one]))
    std = (props - mean) * scale[:, None]
    cov_inv = np.linalg.pinv(np.atleast_2d(_shrunk_covariance(std[:phase_one], None)))
    return _mewma_statistics(std[phase_one:], cov_inv, 0.2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["digits", "image_segments"], required=True)
    ap.add_argument("--unit", type=int, default=5, help="samples per monitoring unit")
    ap.add_argument("--train-fraction", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    x, y = load(args.dataset)
    n_classes = len(np.unique(y))
    classes = [str(c) for c in range(n_classes)]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(x))  # shuffled -> in control by construction
    x, y = x[order], y[order]

    n_train = int(args.train_fraction * len(x))
    x_tr, y_tr = x[:n_train], y[:n_train]
    x_st = x[n_train:]

    scaler = StandardScaler().fit(x_tr)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=args.seed)
    clf.fit(scaler.transform(x_tr), y_tr,
            sample_weight=compute_sample_weight("balanced", y_tr))

    ens = md3.train_random_subspace(
        scaler.transform(x_tr), [str(v) for v in y_tr], classes, seed=args.seed
    )
    ranking = sethi.rank_features(x_tr, [str(v) for v in y_tr], seed=args.seed)

    n_units = len(x_st) // args.unit
    unit_ids = np.array([f"u{i // args.unit:06d}" for i in range(n_units * args.unit)],
                        dtype=object)
    sequence = [f"u{i:06d}" for i in range(n_units)]
    x_st = x_st[: n_units * args.unit]
    phase_one = max(20 * (n_classes - 1), int(0.4 * n_units))
    if n_units - phase_one < 80:
        raise SystemExit(f"only {n_units} units; need a smaller --unit")
    print(f"{args.dataset}: {len(x):,} samples, {n_classes} classes | "
          f"{n_units} units of {args.unit} | Phase I {phase_one} "
          f"| eval {n_units - phase_one}", flush=True)

    def proba(mat):
        p = clf.predict_proba(scaler.transform(mat))
        out = np.zeros((len(mat), n_classes))
        for j, c in enumerate(clf.classes_):
            out[:, int(c)] = p[:, j]
        return out

    def build(mat):
        p = proba(mat)
        frame = composition.build_stream(unit_ids, p, sequence, mode="soft")
        return p, frame

    p_clean, frame_clean = build(x_st)
    coords_clean = composition.ilr_matrix(frame_clean)
    sizes = frame_clean["lot_size"].to_numpy()
    fit = monitor.calibrate(coords_clean[:phase_one], sizes[:phase_one], lam=0.2,
                            target_far=0.005, seed=args.seed)

    def score_set(mat):
        p, frame = build(mat)
        coords = composition.ilr_matrix(frame)
        props = composition.proportion_matrix(frame)
        conf = np.array([p[i * args.unit:(i + 1) * args.unit].max(axis=1).mean()
                         for i in range(n_units)])[phase_one:]
        return {
            "ilr_mewma": monitor.run_monitor(
                fit, coords[phase_one:], sizes[phase_one:])["statistic"],
            "proportion_mewma": proportion_statistic(props, sizes, phase_one),
            "md3_conf": md3.md3_score(
                md3.lot_margin_density(md3.margin_indicator(p), unit_ids, sequence),
                phase_one, lam=0.2),
            "md3_rs": md3.md3_score(
                md3.lot_margin_density(
                    md3.margin_indicator(ens.probabilities(scaler.transform(mat))),
                    unit_ids, sequence),
                phase_one, lam=0.2),
            "adwin_confidence": mc.score_adwin(conf),
            "ks_confidence": mc.score_ks(conf),
        }

    clean = score_set(x_st)
    out = {"dataset": args.dataset, "n_samples": int(len(x)), "n_classes": n_classes,
           "unit": args.unit, "n_units": n_units, "phase_one": phase_one, "arms": {}}

    change_points = [int(phase_one + f * (n_units - phase_one)) for f in
                     (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]

    for which in ("top", "bottom"):
        columns = sethi.select_columns(ranking, which, 0.25)
        q_clean = clf.score(scaler.transform(x_tr), y_tr)
        q_rot = clf.score(scaler.transform(sethi.rotate(x_tr, columns)), y_tr)
        arm = {"n_features_rotated": int(len(columns)),
               "train_accuracy_clean": float(q_clean),
               "train_accuracy_rotated": float(q_rot), "methods": {}}
        print(f"\n{which}-25% ({len(columns)} features): "
              f"train accuracy {q_clean:.3f} -> {q_rot:.3f}", flush=True)

        per_method: dict[str, dict[str, list]] = {}
        for cp in change_points:
            post = np.array([int(u[1:]) >= cp for u in unit_ids])
            deg = sethi.induce(x_st, post, columns)
            scores = score_set(deg)
            k = cp - phase_one
            for name, s in scores.items():
                for target in TARGETS:
                    thr = float(np.quantile(clean[name][k:], 1.0 - target / 1000.0,
                                            method="higher"))
                    hits = np.flatnonzero(s[k:] > thr)
                    per_method.setdefault(name, {}).setdefault(str(target), []).append(
                        float(hits[0]) if len(hits) else float("inf"))

        verdict = "detections" if which == "top" else "FALSE detections (lower better)"
        print(f"{'method':<20s}" + "".join(f"{f'@{t:g}/1000':>20s}" for t in TARGETS)
              + f"   [{verdict}]")
        for name, by in per_method.items():
            cells = []
            arm["methods"][name] = {}
            for target in TARGETS:
                ds = by[str(target)]
                fin = [d for d in ds if np.isfinite(d)]
                arm["methods"][name][str(target)] = {
                    "n_runs": len(ds), "n_detected": len(fin),
                    "median": float(np.median(fin)) if fin else None}
                cells.append(f"{len(fin)}/{len(ds)} med {np.median(fin):.0f}"
                             if fin else f"0/{len(ds)} none")
            print(f"{name:<20s}" + "".join(f"{c:>20s}" for c in cells), flush=True)
        out["arms"][which] = arm

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"second_dataset_{args.dataset}.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
