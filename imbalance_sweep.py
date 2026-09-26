#!/usr/bin/env python3
"""Is compositional monitoring informative only under class imbalance?

An earlier explanation held that ILR-MEWMA works on WM-811K (one
class holds 85% of labelled wafers) and fails on `digits` (ten roughly equal
classes): when classes are balanced, a degrading classifier redistributes its
errors nearly evenly, so the composition of predicted classes barely moves even
as accuracy collapses. Under imbalance the same errors move mass out of a
dominant class and the composition shifts sharply.

That was a conjecture. This tests it by holding everything else fixed ---
dataset, classifier, drift mechanism, protocol, stream length --- and varying
only the share of the stream held by one dominant class.

Two honest caveats, stated because they bound what the experiment can show:

1. Streams are built by sampling with replacement from a held-out pool, because
   `digits` is too small to reach 85% dominance by subsampling alone. Duplicate
   samples therefore appear within a stream. This is a mechanism experiment,
   not a performance benchmark, and the duplication affects every method
   identically.
2. The classifier is trained once, on a balanced split, and frozen. Retraining
   per imbalance level would confound classifier quality with composition
   geometry, which is the thing being measured.

If detection improves monotonically with imbalance, the scope condition is
established. If it does not, the explanation in 6.5 is wrong and should be
withdrawn.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import matched_comparison as mc
from wmmon import composition, md3, monitor, sethi
from wmmon.monitor import _mewma_statistics, _shrunk_covariance

RESULTS = Path(f"{WMMON_HOME}/results")
SHARES = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
UNIT = 3
STREAM_SAMPLES = 1500
HORIZON = 200.0
TARGET = 1.0


def proportion_statistic(props, sizes, phase_one):
    mean = props[:phase_one].mean(axis=0)
    scale = np.sqrt(sizes / np.median(sizes[:phase_one]))
    std = (props - mean) * scale[:, None]
    cov_inv = np.linalg.pinv(np.atleast_2d(_shrunk_covariance(std[:phase_one], None)))
    return _mewma_statistics(std[phase_one:], cov_inv, 0.2)


def load(name):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["digits", "image_segments"],
                    default="digits")
    ap.add_argument("--unit", type=int, default=UNIT)
    args = ap.parse_args()
    t0 = time.time()
    seed = 0
    rng = np.random.default_rng(seed)

    x, y = load(args.dataset)
    n_classes = len(np.unique(y))
    classes = [str(c) for c in range(n_classes)]

    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    n_train = int(0.35 * len(x))
    x_tr, y_tr = x[:n_train], y[:n_train]
    x_pool, y_pool = x[n_train:], y[n_train:]

    scaler = StandardScaler().fit(x_tr)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    clf.fit(scaler.transform(x_tr), y_tr,
            sample_weight=compute_sample_weight("balanced", y_tr))
    ens = md3.train_random_subspace(
        scaler.transform(x_tr), [str(v) for v in y_tr], classes, seed=seed
    )
    ranking = sethi.rank_features(x_tr, [str(v) for v in y_tr], seed=seed)
    columns = sethi.select_columns(ranking, "top", 0.25)
    acc_clean = clf.score(scaler.transform(x_tr), y_tr)
    acc_rot = clf.score(scaler.transform(sethi.rotate(x_tr, columns)), y_tr)
    print(f"classifier frozen: train accuracy {acc_clean:.3f}, "
          f"{acc_rot:.3f} after rotating {len(columns)} features", flush=True)

    dominant = int(np.bincount(y_pool).argmax())
    by_class = {c: np.flatnonzero(y_pool == c) for c in range(n_classes)}

    def proba(mat):
        p = clf.predict_proba(scaler.transform(mat))
        out = np.zeros((len(mat), n_classes))
        for j, c in enumerate(clf.classes_):
            out[:, int(c)] = p[:, j]
        return out

    rows = []
    for share in SHARES:
        draw = np.random.default_rng(seed + int(share * 100))
        n_dom = int(share * STREAM_SAMPLES)
        n_other = STREAM_SAMPLES - n_dom
        per_other = n_other // (n_classes - 1)
        picks = [draw.choice(by_class[dominant], size=n_dom, replace=True)]
        for c in range(n_classes):
            if c == dominant:
                continue
            picks.append(draw.choice(by_class[c], size=per_other, replace=True))
        idx = np.concatenate(picks)
        draw.shuffle(idx)
        x_st = x_pool[idx]

        n_units = len(x_st) // args.unit
        x_st = x_st[: n_units * args.unit]
        unit_ids = np.array([f"u{i // args.unit:06d}" for i in range(len(x_st))],
                            dtype=object)
        sequence = [f"u{i:06d}" for i in range(n_units)]
        phase_one = max(20 * (n_classes - 1), int(0.4 * n_units))

        def score_set(mat):
            p = proba(mat)
            frame = composition.build_stream(unit_ids, p, sequence, mode="soft")
            coords = composition.ilr_matrix(frame)
            props = composition.proportion_matrix(frame)
            sizes = frame["lot_size"].to_numpy()
            conf = np.array([p[i * args.unit:(i + 1) * args.unit].max(axis=1).mean()
                             for i in range(n_units)])[phase_one:]
            fit_local = monitor.calibrate(coords[:phase_one], sizes[:phase_one],
                                          lam=0.2, target_far=0.005, seed=seed)
            return {
                "ilr_mewma": monitor.run_monitor(
                    fit_local, coords[phase_one:], sizes[phase_one:])["statistic"],
                "proportion_mewma": proportion_statistic(props, sizes, phase_one),
                "ks_confidence": mc.score_ks(conf),
                "md3_rs": md3.md3_score(
                    md3.lot_margin_density(
                        md3.margin_indicator(ens.probabilities(scaler.transform(mat))),
                        unit_ids, sequence),
                    phase_one, lam=0.2),
            }, frame

        clean, frame_clean = score_set(x_st)
        realised = composition.proportion_matrix(frame_clean).mean(axis=0)
        change_points = [int(phase_one + f * (n_units - phase_one))
                         for f in (0.2, 0.35, 0.5, 0.65, 0.8)]

        per_method: dict[str, list[float]] = {}
        for cp in change_points:
            post = np.array([int(u[1:]) >= cp for u in unit_ids])
            scores, _ = score_set(sethi.induce(x_st, post, columns))
            k = cp - phase_one
            for name, s in scores.items():
                thr = float(np.quantile(clean[name][k:], 1.0 - TARGET / 1000.0,
                                        method="higher"))
                hits = np.flatnonzero(s[k:] > thr)
                per_method.setdefault(name, []).append(
                    float(hits[0]) if len(hits) else float("inf"))

        row = {"dominant_share_target": share,
               "dominant_share_realised": float(realised.max()),
               "n_units": n_units, "methods": {}}
        for name, delays in per_method.items():
            finite = [x for x in delays if np.isfinite(x)]
            rm = float(np.mean([min(x, HORIZON) if np.isfinite(x) else HORIZON
                                for x in delays]))
            row["methods"][name] = {
                "n_detected": len(finite), "n_runs": len(delays),
                "median": float(np.median(finite)) if finite else None,
                "restricted_mean": rm}
        rows.append(row)
        cells = "  ".join(
            f"{n.split('_')[0]}:{row['methods'][n]['n_detected']}/"
            f"{row['methods'][n]['n_runs']}({row['methods'][n]['restricted_mean']:.0f})"
            for n in ("ilr_mewma", "proportion_mewma", "ks_confidence", "md3_rs"))
        print(f"  dominant {share:.2f} (realised predicted share "
              f"{realised.max():.2f})  {cells}", flush=True)

    print(f"\n{'dominant share':>15s}{'ILR det':>10s}{'ILR RM':>9s}"
          f"{'KS det':>9s}{'KS RM':>8s}{'MD3-RS det':>12s}")
    for r in rows:
        m = r["methods"]
        print(f"{r['dominant_share_target']:>15.2f}"
              f"{m['ilr_mewma']['n_detected']:>7d}/5{m['ilr_mewma']['restricted_mean']:>9.0f}"
              f"{m['ks_confidence']['n_detected']:>6d}/5{m['ks_confidence']['restricted_mean']:>8.0f}"
              f"{m['md3_rs']['n_detected']:>9d}/5")

    shares = [r["dominant_share_realised"] for r in rows]
    ilr_rm = [r["methods"]["ilr_mewma"]["restricted_mean"] for r in rows]
    rho, p = stats.spearmanr(shares, ilr_rm)
    print(f"\nSpearman(dominant share, ILR restricted-mean delay) = "
          f"{rho:+.3f}, p = {p:.4f}")
    print("A strong negative correlation supports the class-imbalance")
    print("explanation: more imbalance, faster compositional detection.")

    RESULTS.mkdir(exist_ok=True)
    json.dump({"dataset": args.dataset, "unit": args.unit, "stream_samples": STREAM_SAMPLES,
               "train_accuracy_clean": float(acc_clean),
               "train_accuracy_rotated": float(acc_rot),
               "spearman_rho": float(rho), "spearman_p": float(p), "rows": rows},
              open(RESULTS / f"imbalance_sweep_{args.dataset}.json", "w"), indent=2)
    print(f"\nwrote results/imbalance_sweep_{args.dataset}.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
