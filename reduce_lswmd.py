"""Load the full LSWMD pickle once and write a compact working subset.

The full file is a 2.1 GB pandas pickle of 811,457 variable-sized wafer maps.
Loading it repeatedly is wasteful and, on a 4 GB machine, close to the limit.
This script loads it exactly once, selects the lots the experiment needs, and
writes a much smaller pickle plus a JSON census of what the full file contains.

The census is computed over ALL rows before subsetting, so the paper can report
true dataset statistics rather than statistics of the subset.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import argparse
import gc
import json
import resource
import sys
from collections import Counter

import numpy as np


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def log(message: str) -> None:
    print(f"[{rss_mb():7.0f} MB] {message}", flush=True)


def scalarise(value):
    while isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        value = value[0]
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=f"{WMMON_HOME}/data/subset.pkl")
    ap.add_argument("--census", default=f"{WMMON_HOME}/data/census.json")
    ap.add_argument("--labelled-lots", type=int, default=900)
    ap.add_argument("--stream-lots", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    log("reading full pickle")
    from legacy_pickle import read_legacy_pickle
    df = read_legacy_pickle(args.data)
    log(f"loaded: {df.shape} columns={list(df.columns)}")

    # --- census over the full file ------------------------------------------
    log("computing census")
    # Keep these as plain object Series of str/None. Casting to the nullable
    # "string" dtype makes every later comparison return pd.NA, which raises on
    # truth-testing -- an error that surfaces well after the expensive load.
    lots = df["lotName"].map(scalarise).astype(object)
    labels = df["failureType"].map(scalarise).astype(object)
    shapes = df["waferMap"].map(
        lambda m: np.asarray(m).shape if m is not None else None
    )

    label_counts = Counter(
        l for l in labels.tolist() if isinstance(l, str) and l.strip() != ""
    )
    wafers_per_lot = lots.value_counts()
    heights = np.array([s[0] for s in shapes if s and len(s) == 2])
    widths = np.array([s[1] for s in shapes if s and len(s) == 2])

    census = {
        "n_wafers": int(len(df)),
        "n_lots": int(lots.nunique()),
        "n_labelled": int(sum(label_counts.values())),
        "label_counts": {k: int(v) for k, v in label_counts.most_common()},
        "wafers_per_lot": {
            "mean": float(wafers_per_lot.mean()),
            "median": float(wafers_per_lot.median()),
            "min": int(wafers_per_lot.min()),
            "max": int(wafers_per_lot.max()),
            "n_lots_with_25": int((wafers_per_lot == 25).sum()),
        },
        "map_dimensions": {
            "height": {
                "min": int(heights.min()),
                "max": int(heights.max()),
                "median": float(np.median(heights)),
            },
            "width": {
                "min": int(widths.min()),
                "max": int(widths.max()),
                "median": float(np.median(widths)),
            },
            "n_distinct_shapes": int(len({s for s in shapes if s})),
        },
    }
    with open(args.census, "w") as fh:
        json.dump(census, fh, indent=2)
    log(f"census written to {args.census}")
    print(json.dumps({k: v for k, v in census.items() if k != "label_counts"}, indent=2))
    print("label counts:", json.dumps(census["label_counts"], indent=2))

    del shapes, heights, widths
    gc.collect()

    # --- lot selection ------------------------------------------------------
    labelled_mask = labels.map(lambda l: isinstance(l, str) and l.strip() != "")
    labelled_lots = sorted(set(lots[labelled_mask].tolist()))
    unlabelled_pool = [l for l in wafers_per_lot.index.tolist() if l not in set(labelled_lots)]

    rng = np.random.default_rng(args.seed)
    pick_labelled = list(
        rng.choice(labelled_lots, size=min(args.labelled_lots, len(labelled_lots)),
                   replace=False)
    )
    pick_stream = list(
        rng.choice(unlabelled_pool, size=min(args.stream_lots, len(unlabelled_pool)),
                   replace=False)
    )
    keep = set(pick_labelled) | set(pick_stream)
    log(f"keeping {len(keep):,} lots "
        f"({len(pick_labelled):,} labelled, {len(pick_stream):,} unlabelled)")

    mask = lots.isin(keep).to_numpy()
    subset = df.loc[mask, ["waferMap", "dieSize", "lotName", "waferIndex", "failureType"]].copy()
    del df
    gc.collect()
    log(f"subset: {subset.shape}")

    subset.to_pickle(args.out)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
