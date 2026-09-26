#!/usr/bin/env python3
"""What happens in lots 1,100-1,500?

The monitor holds its nominal false-alarm rate across the WM-811K stream except
for one burst.  Before that burst can be called a detected process excursion it
has to survive the obvious alternative explanation: that the product changed.
If die size, map geometry or lot size shifts at the same point, the monitor is
reacting to a different product being inspected, not to the inspection module
degrading, and the burst is a confound rather than a result.

This script reports, for the burst window against the Phase I baseline:

  1. which balances carry the statistic (per-coordinate decomposition)
  2. which classes moved, on the simplex, where it is interpretable
  3. whether die size, map shape or lot size moved at the same time
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import json
from collections import Counter
from pathlib import Path

import numpy as np

from wmmon import classifier, composition, data, monitor

CACHE = Path(f"{WMMON_HOME}/cache")
CLASSES = list(data.CLASSES)

#: The Helmert basis used by composition.ilr_basis contrasts the geometric mean
#: of the first i parts against part i+1, so each coordinate has a plain reading.
BALANCE_NAMES = [
    f"g({', '.join(CLASSES[:i])}) vs {CLASSES[i]}" for i in range(1, len(CLASSES))
]


def assembled(expected: int) -> np.ndarray:
    blocks = []
    for path in sorted(CACHE.glob("stream_part_*.npz")):
        with np.load(path) as z:
            blocks.append((int(z["lo"]), z["x"]))
    blocks.sort(key=lambda t: t[0])
    x = np.vstack([b for _, b in blocks])
    assert len(x) == expected, (len(x), expected)
    return x


def main() -> None:
    df, _ = data.load_lswmd(f"{WMMON_HOME}/data/subset.pkl")
    train, _, stream = data.split_labelled(df, seed=0)
    stream = stream.reset_index(drop=True)

    model = classifier.train_inspection_model(
        np.load(CACHE / "train.npz")["x"],
        train["label"].tolist(),
        CLASSES,
        seed=0,
    )
    proba = model.predict_proba(assembled(len(stream)))

    sequence = data.lot_order(stream)
    lots = stream["lot"].to_numpy(dtype=object)
    frame = composition.build_stream(lots, proba, sequence, mode="soft")
    coords = composition.ilr_matrix(frame)
    props = composition.proportion_matrix(frame)
    sizes = frame["lot_size"].to_numpy()

    fit = monitor.calibrate(coords[:300], sizes[:300], lam=0.2, target_far=0.005, seed=0)
    run = monitor.run_monitor(fit, coords[300:], sizes[300:])
    alarm_positions = np.flatnonzero(run["alarm"]) + 300

    print(f"lots: {len(frame):,} | alarms: {len(alarm_positions)}")
    print(f"alarm positions: {alarm_positions.tolist()}\n")

    burst = alarm_positions[(alarm_positions >= 1100) & (alarm_positions < 1500)]
    print(f"burst window 1100-1500 holds {len(burst)} of them\n")

    results: dict = {
        "n_lots": int(len(frame)),
        "alarm_positions": alarm_positions.tolist(),
        "burst_positions": burst.tolist(),
    }

    # --- 1. which balances carry the statistic --------------------------------
    if len(burst):
        contributions = np.vstack(
            [monitor.diagnose(fit, coords, sizes, int(p)) for p in burst]
        )
        mean_contribution = contributions.mean(axis=0)
        order = np.argsort(-np.abs(mean_contribution))
        print("balance contributions at burst alarms (mean over alarms):")
        for j in order[:4]:
            share = 100 * mean_contribution[j] / mean_contribution.sum()
            print(f"  ilr{j}  {mean_contribution[j]:9.2f}  ({share:5.1f}%)  {BALANCE_NAMES[j]}")
        results["balance_contributions"] = {
            f"ilr{j}": {
                "value": float(mean_contribution[j]),
                "share": float(mean_contribution[j] / mean_contribution.sum()),
                "reads_as": BALANCE_NAMES[j],
            }
            for j in order
        }
        print()

    # --- 2. which classes moved ----------------------------------------------
    baseline = props[:300].mean(axis=0)
    window = props[1100:1500].mean(axis=0)
    print("mean predicted-class composition (Phase I -> burst window):")
    shifts = {}
    for j, name in enumerate(CLASSES):
        ratio = window[j] / baseline[j] if baseline[j] > 0 else np.inf
        shifts[name] = {
            "phase1": float(baseline[j]),
            "burst": float(window[j]),
            "fold_change": float(ratio),
        }
        print(f"  {name:<11s} {baseline[j]:7.4f} -> {window[j]:7.4f}   x{ratio:5.2f}")
    results["class_shifts"] = shifts
    print()

    # --- 3. is it a product change? ------------------------------------------
    lot_meta = (
        stream.assign(shape=stream["wafer_map"].map(lambda m: np.asarray(m).shape))
        .groupby("lot")
        .agg(die_size=("die_size", "median"), shape=("shape", lambda s: s.mode().iloc[0]))
    )
    lot_meta = lot_meta.loc[[l for l in frame["lot"]]]
    lot_meta["position"] = np.arange(len(lot_meta))

    def describe(lo: int, hi: int, label: str) -> dict:
        block = lot_meta[(lot_meta["position"] >= lo) & (lot_meta["position"] < hi)]
        block_sizes = sizes[lo:hi]
        shape_counts = Counter(block["shape"])
        top_shape, top_n = shape_counts.most_common(1)[0]
        out = {
            "n_lots": int(len(block)),
            "die_size_median": float(block["die_size"].median()),
            "die_size_iqr": float(
                block["die_size"].quantile(0.75) - block["die_size"].quantile(0.25)
            ),
            "n_distinct_shapes": int(len(shape_counts)),
            "dominant_shape": str(top_shape),
            "dominant_shape_share": float(top_n / len(block)),
            "lot_size_median": float(np.median(block_sizes)),
        }
        print(
            f"  {label:<22s} die {out['die_size_median']:8.1f} (IQR {out['die_size_iqr']:7.1f}) "
            f"| shapes {out['n_distinct_shapes']:>4d} | top {out['dominant_shape']:<10s} "
            f"{100 * out['dominant_shape_share']:5.1f}% | lot size {out['lot_size_median']:4.1f}"
        )
        return out

    print("product characteristics by window:")
    results["windows"] = {
        "phase_one_0_300": describe(0, 300, "Phase I (0-300)"),
        "quiet_700_1100": describe(700, 1100, "quiet (700-1100)"),
        "burst_1100_1500": describe(1100, 1500, "BURST (1100-1500)"),
        "quiet_1500_1900": describe(1500, 1900, "quiet (1500-1900)"),
    }

    Path(f"{WMMON_HOME}/results").mkdir(parents=True, exist_ok=True)
    with open(f"{WMMON_HOME}/results/burst_analysis.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote results/burst_analysis.json")


if __name__ == "__main__":
    main()
