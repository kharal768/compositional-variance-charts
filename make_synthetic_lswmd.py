#!/usr/bin/env python3
"""Generate a small pickle with the exact LSWMD schema, for smoke-testing only.

This is NOT data for the paper. It exists so the pipeline can be verified to
run end to end before the real 214 MB file is downloaded. Any number produced
from it is meaningless and must never appear in a manuscript.
"""
import argparse
import numpy as np
import pandas as pd

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
           "Near-full", "Random", "Scratch", "none"]
# Roughly the real class balance: 'none' dominates.
PRIOR = np.array([0.035, 0.005, 0.05, 0.055, 0.04, 0.001, 0.005, 0.009, 0.80])


def make_map(cls, rng, size=32):
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (size / 2)
    wafer = np.where(r <= 1.0, 1, 0).astype(np.uint8)
    valid = wafer > 0
    fail = rng.random((size, size)) < 0.02  # background failures
    if cls == "Center":
        fail |= (r < 0.3) & (rng.random((size, size)) < 0.7)
    elif cls == "Donut":
        fail |= (r > 0.35) & (r < 0.6) & (rng.random((size, size)) < 0.7)
    elif cls == "Edge-Ring":
        fail |= (r > 0.8) & (rng.random((size, size)) < 0.8)
    elif cls == "Edge-Loc":
        ang = np.arctan2(yy - cy, xx - cx)
        fail |= (r > 0.75) & (np.abs(ang - rng.uniform(-3, 3)) < 0.6)
    elif cls == "Loc":
        py, px = rng.integers(8, size - 8, 2)
        fail |= (np.abs(yy - py) < 4) & (np.abs(xx - px) < 4)
    elif cls == "Scratch":
        slope = rng.uniform(-1.5, 1.5)
        fail |= np.abs((yy - cy) - slope * (xx - cx)) < 1.2
    elif cls == "Random":
        fail |= rng.random((size, size)) < 0.25
    elif cls == "Near-full":
        fail |= rng.random((size, size)) < 0.85
    wafer[valid & fail] = 2
    return wafer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lots", type=int, default=900)
    ap.add_argument("--out", default="synthetic_lswmd.pkl")
    ap.add_argument("--labelled-fraction", type=float, default=0.21)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    rows = []
    for lot_id in range(1, a.lots + 1):
        n_wafers = int(rng.choice([25, 25, 25, 24, 23, 12], p=[.55, .15, .1, .1, .06, .04]))
        labelled_lot = rng.random() < a.labelled_fraction
        for widx in range(1, n_wafers + 1):
            cls = rng.choice(CLASSES, p=PRIOR)
            wmap = make_map(cls, rng)
            rows.append({
                "waferMap": wmap,
                "dieSize": float((wmap > 0).sum()),
                "lotName": f"lot{lot_id}",
                "waferIndex": float(widx),
                "trainTestLabel": [["Training"]],
                "failureType": [[cls]] if labelled_lot else [],
            })
    pd.DataFrame(rows).to_pickle(a.out)
    print(f"wrote {a.out}: {len(rows):,} wafers, {a.lots:,} lots")


if __name__ == "__main__":
    main()
