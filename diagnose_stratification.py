#!/usr/bin/env python3
"""Why is the in-control false-alarm rate 11x nominal on real WM-811K?

The first run on real data produced 55 false alarms per 1,000 lots against a
nominal 5, with no degradation injected.  Before any detection delay from this
pipeline means anything, that has to be explained.  The hypothesis under test:
the raw lot sequence is not one process.  Consecutive lots are different
products with different die counts and map geometries, so the predicted-class
composition moves for reasons unrelated to inspection-module health.

If that is right, stratifying by die size --- monitoring within a product
family, which is what a fab actually does --- should bring the realised rate
back toward nominal.  If it is not right, the premise of the paper is in
trouble and it is better to know now.

Features are cached to disk so the expensive extraction happens once.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from wmmon import classifier, composition, data, monitor
from wmmon import features as feat


def scalar(value):
    while isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        value = value[0]
    return value


def cached_features(maps, cache: Path) -> np.ndarray:
    if cache.exists():
        print(f"  using cached features {cache.name}", flush=True)
        return np.load(cache)["x"]
    x = feat.extract_batch(maps, progress_every=2000, n_jobs=1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x)
    return x


def assembled_stream_features(cache: Path, expected: int) -> np.ndarray:
    """Reassemble the blocks written by extract_stream.py, in index order."""
    parts = sorted(cache.glob("stream_part_*.npz"))
    if not parts:
        raise FileNotFoundError("run extract_stream.py first")
    blocks = []
    for path in parts:
        with np.load(path) as z:
            blocks.append((int(z["lo"]), z["x"]))
    blocks.sort(key=lambda t: t[0])
    x = np.vstack([b for _, b in blocks])
    if len(x) != expected:
        raise ValueError(f"assembled {len(x)} rows, expected {expected}")
    print(f"  assembled {len(parts)} cached blocks -> {x.shape}", flush=True)
    return x


def evaluate_stream(coords, sizes, phase_one, label, target_far=0.005):
    """Calibrate on the first `phase_one` lots, measure realised FAR on the rest."""
    if len(coords) < phase_one + 100:
        return None
    fit = monitor.calibrate(
        coords[:phase_one], sizes[:phase_one], lam=0.2, target_far=target_far, seed=0
    )
    run = monitor.run_monitor(fit, coords[phase_one:], sizes[phase_one:])
    m = monitor.detection_metrics(run["alarm"], None, warmup=20)
    out = {
        "label": label,
        "n_lots": int(len(coords)),
        "phase_one": phase_one,
        "limit": fit.limit,
        "prewhitened": fit.phi is not None,
        "lag1_dependence": fit.lag1_dependence,
        "far_per_1000": m["far_per_1000_lots"],
    }
    print(
        f"  {label:<34s} lots={out['n_lots']:>5d} rho={out['lag1_dependence']:.2f} "
        f"phi={'y' if out['prewhitened'] else 'n'} limit={out['limit']:6.2f} "
        f"FAR/1000={out['far_per_1000']:7.2f}",
        flush=True,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{WMMON_HOME}/data/subset.pkl")
    ap.add_argument("--cache", default=f"{WMMON_HOME}/cache")
    ap.add_argument("--out", default=f"{WMMON_HOME}/results/stratification.json")
    ap.add_argument("--phase-one", type=int, default=300)
    args = ap.parse_args()

    cache = Path(args.cache)
    results = {"target_far_per_1000": 5.0}

    print("[1/4] loading subset")
    df, report = data.load_lswmd(args.data)
    df["die_size_key"] = df["die_size"].round().astype("Int64")
    print(f"  {len(df):,} wafers | {df['lot'].nunique():,} lots")

    print("[2/4] training inspection module")
    train, test, stream = data.split_labelled(df, seed=0)
    x_train = cached_features(train["wafer_map"].tolist(), cache / "train.npz")
    model = classifier.train_inspection_model(
        x_train, train["label"].tolist(), list(data.CLASSES), seed=0
    )

    print("[3/4] extracting stream features (cached)")
    stream = stream.reset_index(drop=True)
    x_stream = assembled_stream_features(cache, len(stream))
    proba = model.predict_proba(x_stream)

    # --- how heterogeneous is the stream? -----------------------------------
    lot_die = stream.groupby("lot")["die_size_key"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else pd.NA
    )
    lot_shape = stream.groupby("lot")["wafer_map"].agg(
        lambda s: Counter(np.asarray(m).shape for m in s).most_common(1)[0][0]
    )
    die_counts = lot_die.value_counts()
    results["heterogeneity"] = {
        "n_distinct_die_sizes": int(lot_die.nunique()),
        "n_distinct_map_shapes": int(lot_shape.nunique()),
        "top_die_sizes": {str(k): int(v) for k, v in die_counts.head(10).items()},
        "share_in_largest_family": float(die_counts.iloc[0] / die_counts.sum()),
    }
    print(f"  distinct die sizes across lots : {lot_die.nunique()}")
    print(f"  distinct map shapes across lots: {lot_shape.nunique()}")
    print(f"  largest family holds {100 * die_counts.iloc[0] / die_counts.sum():.1f}% of lots")

    print("[4/4] in-control comparison")
    runs = []

    # (a) raw sequence, all products mixed -- the configuration that gave 55/1000
    sequence = data.lot_order(stream)
    frame = composition.build_stream(
        stream["lot"].to_numpy(dtype=object), proba, sequence, mode="soft"
    )
    runs.append(
        evaluate_stream(
            composition.ilr_matrix(frame),
            frame["lot_size"].to_numpy(),
            args.phase_one,
            "raw sequence (all products)",
        )
    )

    # (b) within each large die-size family
    for die_size, n_lots in die_counts.head(6).items():
        if n_lots < args.phase_one + 150:
            continue
        family_lots = set(lot_die[lot_die == die_size].index)
        sub = stream[stream["lot"].isin(family_lots)]
        idx = sub.index.to_numpy()
        sub_sequence = [l for l in data.lot_order(sub)]
        sub_frame = composition.build_stream(
            sub["lot"].to_numpy(dtype=object), proba[idx], sub_sequence, mode="soft"
        )
        runs.append(
            evaluate_stream(
                composition.ilr_matrix(sub_frame),
                sub_frame["lot_size"].to_numpy(),
                args.phase_one,
                f"die size {die_size} only",
            )
        )

    results["runs"] = [r for r in runs if r]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
