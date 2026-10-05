#!/usr/bin/env python3
"""Extract stream features in resumable blocks.

Background jobs in this environment get killed without warning, and a single
uninterrupted pass over 47,281 wafers is long enough to be caught by it.  This
script writes each block to its own file and skips blocks already on disk, so
it can be run repeatedly until it completes and never repeats work.

Single-process by design: forking a worker doubles peak memory through
copy-on-write page faults for no speed gain on one core.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import argparse
import time
from pathlib import Path

import numpy as np

from wmmon import data
from wmmon import features as feat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{WMMON_HOME}/data/subset.pkl")
    ap.add_argument("--cache", default=f"{WMMON_HOME}/cache")
    ap.add_argument("--block", type=int, default=4000)
    ap.add_argument("--budget", type=int, default=250, help="seconds before stopping")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    df, _ = data.load_lswmd(args.data)
    _, _, stream = data.split_labelled(df, seed=0)
    stream = stream.reset_index(drop=True)
    maps = stream["wafer_map"].tolist()
    del df

    n = len(maps)
    n_blocks = (n + args.block - 1) // args.block
    start = time.time()
    done = 0

    for b in range(n_blocks):
        path = cache / f"stream_part_{b:03d}.npz"
        if path.exists():
            done += 1
            continue
        if time.time() - start > args.budget:
            print(f"budget reached; {done}/{n_blocks} blocks present", flush=True)
            return
        lo, hi = b * args.block, min((b + 1) * args.block, n)
        x = feat.extract_batch(maps[lo:hi], progress_every=0, n_jobs=1)
        np.savez_compressed(path, x=x, lo=lo, hi=hi)
        done += 1
        print(f"block {b:03d} [{lo}:{hi}] -> {path.name} "
              f"({time.time() - start:.0f}s elapsed)", flush=True)

    print(f"all {n_blocks} blocks present", flush=True)


if __name__ == "__main__":
    main()
