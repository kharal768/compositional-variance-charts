#!/usr/bin/env python3
"""Detection delay across the whole false-alarm range, not at one point.

Two problems with the previous comparison are fixed here.

**The calibration leak.**  Thresholds were chosen on the same undegraded lots
the false-alarm rate was then quoted from, which flatters every method.  Here
the threshold is fitted on clean lots [change_point, end) --- in control in the
undegraded stream, and disjoint from the window where false alarms are counted
--- and the rate is reported on lots [warmup, change_point).

**The single operating point.**  A method can win at one false-alarm budget and
lose across the rest of the range.  A fab does not run at 5 per 1,000 because a
paper did; it runs wherever its alarm-handling capacity sits.  Sweeping the
budget shows whether any advantage is real or an artefact of where the line was
drawn.

    python operating_curve.py --mode defect_remix --confine-to-defective
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import argparse
import json
import time
from pathlib import Path

import numpy as np

import matched_comparison as mc
from wmmon import baselines, classifier, composition, data, monitor

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
CLASSES = list(data.CLASSES)

TARGETS_PER_1000 = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--confine-to-defective", action="store_true")
    ap.add_argument("--phase-one", type=int, default=300)
    ap.add_argument("--stream-lots", type=int, default=1200)
    ap.add_argument("--change-point", type=int, default=800)
    ap.add_argument("--ramp-lots", type=int, default=0)
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    df, _ = data.load_lswmd(f"{WMMON_HOME}/data/subset.pkl")
    train, _, stream = data.split_labelled(df, seed=args.seed)
    stream = stream.reset_index(drop=True)
    del df

    model = classifier.train_inspection_model(
        np.load(CACHE / "train.npz")["x"], train["label"].tolist(), CLASSES,
        seed=args.seed,
    )

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
    lots = lots_all[rows]
    position_of = {lot: i for i, lot in enumerate(sequence)}
    post = np.array([position_of[str(l)] for l in lots]) >= args.change_point

    confine = "_conf" if args.confine_to_defective else ""
    # The change point and seed determine WHICH wafers are degraded and how, so
    # both belong in the cache key. Without them a rerun at a different change
    # point silently reuses the wrong features whenever the array lengths happen
    # to match, and the run looks clean.
    cache_path = CACHE / (
        f"deg_{args.mode}{confine}_s{args.severity}_r{args.ramp_lots}"
        f"_cp{args.change_point}_seed{args.seed}.npz"
    )
    if not cache_path.exists():
        raise FileNotFoundError(f"{cache_path.name} missing; run run_degradation.py first")

    x_clean = x_all[rows]
    x_deg = x_clean.copy()
    x_deg[post] = np.load(cache_path)["x"]

    off, warmup = args.phase_one, 20
    cp = args.change_point - off

    proba_clean, frame_clean = mc.build(model, x_clean, lots, sequence)
    proba_deg, frame_deg = mc.build(model, x_deg, lots, sequence)
    coords_clean = composition.ilr_matrix(frame_clean)
    coords_deg = composition.ilr_matrix(frame_deg)
    props_clean = composition.proportion_matrix(frame_clean)
    props_deg = composition.proportion_matrix(frame_deg)
    sizes = frame_clean["lot_size"].to_numpy()

    fit = monitor.calibrate(coords_clean[:off], sizes[:off], lam=0.2,
                            target_far=0.005, seed=args.seed)
    conf_clean = baselines.max_confidence_stream(proba_clean, lots, sequence)[off:]
    conf_deg = baselines.max_confidence_stream(proba_deg, lots, sequence)[off:]

    scorers = {
        "ilr_mewma": lambda c, p, s: monitor.run_monitor(
            fit, c[off:], sizes[off:])["statistic"],
        "proportion_mewma": lambda c, p, s: mc._proportion_statistic(p, sizes, off),
        "page_hinkley_confidence": lambda c, p, s: mc.score_page_hinkley(s),
        "adwin_confidence": lambda c, p, s: mc.score_adwin(s),
        "ks_confidence": lambda c, p, s: mc.score_ks(s),
        "mmd_ilr": lambda c, p, s: mc.score_mmd(c[off:], seed=args.seed),
    }

    print(f"calibration window: clean lots [{cp}, end)  "
          f"| false-alarm window: lots [{warmup}, {cp})  "
          f"| delay measured from lot {cp}\n", flush=True)

    out = {
        "mode": args.mode,
        "confined": bool(args.confine_to_defective),
        "ramp_lots": args.ramp_lots,
        "targets_per_1000": TARGETS_PER_1000,
        "methods": {},
    }

    header = f"{'method':<26s}" + "".join(f"{t:>8.0f}" for t in TARGETS_PER_1000)
    print(f"{'':<26s}{'target false alarms per 1,000 lots':>48s}")
    print(header)
    print(f"{'':<26s}" + "".join(f"{'delay':>8s}" for _ in TARGETS_PER_1000))

    for name, fn in scorers.items():
        clean_scores = fn(coords_clean, props_clean, conf_clean)
        deg_scores = fn(coords_deg, props_deg, conf_deg)

        entry = {}
        cells = []
        for target in TARGETS_PER_1000:
            rate = target / 1000.0
            # Threshold from the held-out clean segment.
            calib = clean_scores[cp:]
            threshold = (
                float(np.quantile(calib, 1.0 - rate, method="higher"))
                if len(calib) else np.inf
            )
            # False alarms on the disjoint pre-change window.
            fa_window = deg_scores[warmup:cp] > threshold
            fa_lots = int(fa_window.sum())
            fa_rate = 1000.0 * fa_lots / max(cp - warmup, 1)

            hits = np.flatnonzero(deg_scores[cp:] > threshold)
            delay = float(hits[0]) if len(hits) else float("inf")

            entry[str(target)] = {
                "threshold": threshold,
                "realised_fa_per_1000": fa_rate,
                "detection_delay": delay,
            }
            cells.append("never" if not np.isfinite(delay) else f"{delay:.0f}")

        out["methods"][name] = entry
        print(f"{name:<26s}" + "".join(f"{c:>8s}" for c in cells), flush=True)

    print("\nrealised false-alarm rates (per 1,000, pre-change window):")
    for name, entry in out["methods"].items():
        rates = "".join(f"{entry[str(t)]['realised_fa_per_1000']:>8.1f}"
                        for t in TARGETS_PER_1000)
        print(f"{name:<26s}{rates}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = confine + (f"_ramp{args.ramp_lots}" if args.ramp_lots else "")
    path = RESULTS / f"curve_{args.mode}{suffix}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
