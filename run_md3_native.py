#!/usr/bin/env python3
"""Run MD3 under its own threshold rule rather than this study's.

The Sethi false-alarm experiment showed MD3-RS firing on 12 of 12 replications
of a rotation that costs two points of macro-F1 --- the exact behaviour its
authors criticise HDDDM for. Before that goes in a paper it has to be checked,
because the MD3 in this study is calibrated with the study-wide empirical
threshold, not the rule from the paper. If the failure is an artefact of my
calibration choice it is a finding about me, not about MD3.

The native rule (their Eq. 9): signal when

    |MD_t - MD_ref| > theta * sigma_ref

with the reference mean and standard deviation estimated by K-fold
cross-validation on the training set, theta suggested in [0, 3] and 2 used
throughout their experiments, and MD_t an EWMA over samples with forgetting
factor lambda = (N-1)/N for chunk size N.

One adaptation, stated because it matters: their sigma_ref is the spread of MD
across K cross-validation folds, each containing a large slice of the training
set. The monitoring unit here is a lot of ~25 wafers, whose MD is far more variable
than a fold's. Estimating sigma_ref from folds and then applying it to lots
would compare a lot-scale deviation against a fold-scale spread and guarantee
alarms. So sigma_ref is estimated at the scale the chart actually runs at, by
resampling training subsets of the stream's own chunk size. Both versions are
reported below, because the difference between them is the point.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

import matched_comparison as mc
from wmmon import classifier, data, md3, sethi

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
CLASSES = list(data.CLASSES)
CHANGE_POINTS = [450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
PHASE_ONE = 300


def reference_by_folds(x, y, seed: int, n_splits: int = 5) -> tuple[float, float]:
    """MD_ref and sigma_ref from K-fold CV, as written in the paper."""
    values = []
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x):
        ens = md3.train_random_subspace(x[tr], [y[i] for i in tr], CLASSES, seed=seed)
        values.append(float(md3.margin_indicator(ens.probabilities(x[te])).mean()))
    return float(np.mean(values)), float(np.std(values, ddof=1))


def out_of_fold_indicator(x, y, seed: int, n_splits: int = 5) -> np.ndarray:
    """Held-out margin indicator for every training wafer.

    In-sample predictions are useless here: the base learners are unpruned
    decision trees, so they fit the training set exactly, every wafer sits far
    from the margin, and the estimated margin density is identically zero. The
    reference has to come from data the ensemble did not see --- which is
    precisely why the original method estimates it by cross-validation.
    """
    indicator = np.zeros(len(x))
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x):
        ens = md3.train_random_subspace(x[tr], [y[i] for i in tr], CLASSES, seed=seed)
        indicator[te] = md3.margin_indicator(ens.probabilities(x[te]))
    return indicator


def reference_by_chunks(
    indicator: np.ndarray, chunk: int, n_draws: int, seed: int
) -> tuple[float, float]:
    """MD_ref and sigma_ref estimated at the chart's own chunk size."""
    rng = np.random.default_rng(seed)
    draws = [
        float(indicator[rng.choice(len(indicator), size=chunk, replace=False)].mean())
        for _ in range(n_draws)
    ]
    return float(np.mean(draws)), float(np.std(draws, ddof=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=2.0)
    ap.add_argument("--stream-lots", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    df, _ = data.load_lswmd(f"{WMMON_HOME}/data/subset.pkl")
    train, _, stream = data.split_labelled(df, seed=args.seed)
    stream = stream.reset_index(drop=True)
    x_train = np.load(CACHE / "train.npz")["x"]
    y_train = train["label"].tolist()

    ens = md3.train_random_subspace(x_train, y_train, CLASSES, seed=args.seed)
    model = classifier.train_inspection_model(x_train, y_train, CLASSES, seed=args.seed)

    blocks = []
    for path in sorted(CACHE.glob("stream_part_*.npz")):
        with np.load(path) as z:
            blocks.append((int(z["lo"]), z["x"]))
    blocks.sort(key=lambda t: t[0])
    x_all = np.vstack([b for _, b in blocks])

    sequence = data.lot_order(stream)[: args.stream_lots]
    keep = set(sequence)
    lots_all = stream["lot"].to_numpy(dtype=object)
    rows = np.flatnonzero(np.array([str(l) in keep for l in lots_all]))
    lots, x_clean = lots_all[rows], x_all[rows]
    position = np.array([{l: i for i, l in enumerate(sequence)}[str(l)] for l in lots])

    _, frame = mc.build(model, x_clean, lots, sequence)
    chunk = int(np.median(frame["lot_size"].to_numpy()))

    ref_f, sd_f = reference_by_folds(x_train, y_train, args.seed)
    oof = out_of_fold_indicator(x_train, y_train, args.seed)
    ref_c, sd_c = reference_by_chunks(oof, chunk, 2000, args.seed)
    print(f"chunk size (median lot) = {chunk}")
    print(f"fold-scale reference : MD_ref {ref_f:.4f}  sigma {sd_f:.4f}  "
          f"limit {args.theta * sd_f:.4f}")
    print(f"chunk-scale reference: MD_ref {ref_c:.4f}  sigma {sd_c:.4f}  "
          f"limit {args.theta * sd_c:.4f}\n", flush=True)

    order = sethi.rank_features(x_train, y_train, seed=args.seed)

    def run(which: str | None):
        """Alarms per change point for one experiment; None = undegraded."""
        results = {"folds": [], "chunks": []}
        columns = None if which is None else sethi.select_columns(order, which, 0.25)
        for cp_lot in CHANGE_POINTS:
            post = position >= cp_lot
            x_deg = x_clean if columns is None else sethi.induce(x_clean, post, columns)
            density = md3.lot_margin_density(
                md3.margin_indicator(ens.probabilities(x_deg)), lots, sequence
            )
            score = md3.md3_score(density, PHASE_ONE, lam=0.2)
            cp = cp_lot - PHASE_ONE
            for tag, sd in (("folds", sd_f), ("chunks", sd_c)):
                alarm = score > args.theta * sd
                hits = np.flatnonzero(alarm[cp:])
                pre = int(alarm[20:cp].sum())
                results[tag].append(
                    {"change_point": cp_lot,
                     "delay": float(hits[0]) if len(hits) else float("inf"),
                     "pre_change_alarm_lots": pre,
                     "pre_change_rate_per_1000":
                         1000.0 * pre / max(cp - 20, 1)})
            if columns is None:
                break  # in-control run does not depend on the change point
        return results

    out = {"theta": args.theta, "chunk": chunk,
           "reference": {"folds": [ref_f, sd_f], "chunks": [ref_c, sd_c]}}

    for label, which in (("in_control", None), ("top25", "top"), ("bottom25", "bottom")):
        res = run(which)
        out[label] = res
        for tag in ("folds", "chunks"):
            runs = res[tag]
            det = [r for r in runs if np.isfinite(r["delay"])]
            rate = float(np.mean([r["pre_change_rate_per_1000"] for r in runs]))
            if label == "in_control":
                print(f"{label:<11s} {tag:<7s} in-control alarm rate "
                      f"{rate:7.2f} per 1,000 lots", flush=True)
            else:
                med = np.median([r["delay"] for r in det]) if det else float("nan")
                print(f"{label:<11s} {tag:<7s} fires {len(det)}/{len(runs)}  "
                      f"median delay {med:.0f}  pre-change rate {rate:6.2f}/1,000",
                      flush=True)

    RESULTS.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS / "md3_native_threshold.json", "w"), indent=2,
              default=str)
    print(f"\nwrote results/md3_native_threshold.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
