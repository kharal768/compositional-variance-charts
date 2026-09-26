#!/usr/bin/env python3
"""Reconcile two conflicting in-control false-alarm figures.

The pipeline reported 55.08 false alarms per 1,000 lots; the stratification
diagnostic reported 9.84.  Both used the same data, the same seed and a Phase I
of the first 300 lots, so at most one of them can be a property of the method.
This script isolates the difference by running the two configurations side by
side and printing the Phase I inputs, not just the outputs.

Reporting the lower number without explaining the higher one would be choosing
the flattering result.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import json
from pathlib import Path

import numpy as np

from wmmon import classifier, composition, data, monitor

CACHE = Path(f"{WMMON_HOME}/cache")


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

    x_train = np.load(CACHE / "train.npz")["x"]
    model = classifier.train_inspection_model(
        x_train, train["label"].tolist(), list(data.CLASSES), seed=0
    )
    proba = model.predict_proba(assembled(len(stream)))
    full_sequence = data.lot_order(stream)
    lots = stream["lot"].to_numpy(dtype=object)

    print(f"stream lots available: {len(full_sequence):,}\n")
    report = {}

    for label, n_lots in [
        ("pipeline config (1,500 lots)", 1500),
        ("diagnostic config (all lots)", len(full_sequence)),
    ]:
        sequence = full_sequence[:n_lots]
        keep = set(sequence)
        mask = np.array([str(l) in keep for l in lots])
        frame = composition.build_stream(
            lots[mask], proba[mask], sequence, mode="soft"
        )
        coords = composition.ilr_matrix(frame)
        sizes = frame["lot_size"].to_numpy()

        fit = monitor.calibrate(coords[:300], sizes[:300], lam=0.2,
                                target_far=0.005, seed=0)
        run = monitor.run_monitor(fit, coords[300:], sizes[300:])
        m = monitor.detection_metrics(run["alarm"], None, warmup=20)

        # Phase I fingerprint: if these differ, the streams differ.
        phase1 = coords[:300]
        entry = {
            "lots_in_stream": int(len(frame)),
            "phase1_mean_norm": float(np.linalg.norm(phase1.mean(axis=0))),
            "phase1_total_var": float(phase1.var(axis=0).sum()),
            "phase1_median_lot_size": float(np.median(sizes[:300])),
            "lag1_dependence": fit.lag1_dependence,
            "prewhitened": fit.phi is not None,
            "limit": fit.limit,
            "eval_lots": int(len(coords) - 300),
            "far_per_1000": m["far_per_1000_lots"],
        }
        report[label] = entry
        print(f"{label}")
        for k, v in entry.items():
            print(f"    {k:<22s} {v}")
        print()

    a, b = report.values()
    same_phase1 = (
        abs(a["phase1_mean_norm"] - b["phase1_mean_norm"]) < 1e-9
        and abs(a["phase1_total_var"] - b["phase1_total_var"]) < 1e-9
    )
    print(f"Phase I identical across configs: {same_phase1}")
    if same_phase1 and abs(a["limit"] - b["limit"]) > 1e-6:
        print("  -> same Phase I, different limit: calibration is not deterministic.")
    if same_phase1 and abs(a["limit"] - b["limit"]) < 1e-6:
        print("  -> same limit; the gap is the evaluation window, not calibration.")

    # Where in the stream do the alarms sit? A rate that rises with lot index
    # means drift, not miscalibration.
    sequence = full_sequence
    keep = set(sequence)
    mask = np.array([str(l) in keep for l in lots])
    frame = composition.build_stream(lots[mask], proba[mask], sequence, mode="soft")
    coords = composition.ilr_matrix(frame)
    sizes = frame["lot_size"].to_numpy()
    fit = monitor.calibrate(coords[:300], sizes[:300], lam=0.2, target_far=0.005, seed=0)
    alarm = monitor.run_monitor(fit, coords[300:], sizes[300:])["alarm"]

    print("\nalarm rate by position in the stream (per 1,000 lots):")
    chunk = 400
    profile = []
    for i in range(0, len(alarm), chunk):
        window = alarm[i : i + chunk]
        rate = 1000.0 * window.mean()
        profile.append({"from_lot": 300 + i, "n": int(len(window)), "rate": float(rate)})
        print(f"    lots {300 + i:>5d}-{300 + i + len(window):>5d}  {rate:8.2f}")
    report["alarm_profile"] = profile

    with open(f"{WMMON_HOME}/results/reconciliation.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote results/reconciliation.json")


if __name__ == "__main__":
    main()
