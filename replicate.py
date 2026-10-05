#!/usr/bin/env python3
"""Replicate the operating-curve experiment across change points.

Every delay reported so far comes from a single run with the excursion starting
at lot 800.  A difference of 92 versus 93 lots is not a difference until it
survives replication, and reporting one run as if it were an estimate is how
tables get published that nobody can reproduce.

Replication here moves the change point through the stream.  That varies which
lots are degraded, how much in-control history precedes the excursion, and
which products are affected --- the sources of variability a fab would actually
see.  It is a stronger replication than reseeding a random transform, which
leaves the deterministic modes (resolution loss, edge exclusion) with exactly
zero variance and would understate the spread.

    python replicate.py --mode defect_remix --confine-to-defective

Each change point needs its own feature extraction (~45 s), so run one mode per
invocation and let the cache accumulate.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import argparse
import json
import time
from pathlib import Path

import numpy as np

import matched_comparison as mc
from wmmon import baselines, classifier, composition, data, drift, md3, monitor
from wmmon import features as feat

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
CLASSES = list(data.CLASSES)

DEFAULT_CHANGE_POINTS = [450, 500, 550, 600, 650, 700, 750, 800,
                         850, 900, 950, 1000]
TARGETS_PER_1000 = [1.0, 5.0, 20.0]


def degraded_features(stream, rows, lots, sequence, post, args, change_point,
                      proba_clean):
    confine = "_conf" if args.confine_to_defective else ""
    path = CACHE / (
        f"deg_{args.mode}{confine}_s{args.severity}_r{args.ramp_lots}"
        f"_cp{change_point}_seed{args.seed}.npz"
    )
    if path.exists():
        return np.load(path)["x"]

    maps_post = [np.asarray(m) for m in stream["wafer_map"].to_numpy()[rows[post]]]
    degraded = drift.apply_degradation(
        maps_post, lots[post], sequence, args.mode,
        change_point=change_point, max_severity=args.severity,
        ramp_lots=args.ramp_lots, seed=args.seed,
    )
    if args.confine_to_defective:
        normal = proba_clean[post].argmax(axis=1) == CLASSES.index("none")
        for i in np.flatnonzero(normal):
            degraded[i] = maps_post[i]
    x = feat.extract_batch(degraded, progress_every=0, n_jobs=1)
    np.savez_compressed(path, x=x)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--confine-to-defective", action="store_true")
    ap.add_argument("--phase-one", type=int, default=300)
    ap.add_argument("--stream-lots", type=int, default=1200)
    ap.add_argument("--ramp-lots", type=int, default=0)
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--change-points", type=int, nargs="*",
                    default=DEFAULT_CHANGE_POINTS)
    ap.add_argument("--budget-seconds", type=int, default=240)
    args = ap.parse_args()
    t0 = time.time()

    df, _ = data.load_lswmd(f"{WMMON_HOME}/data/subset.pkl")
    train, _, stream = data.split_labelled(df, seed=args.seed)
    stream = stream.reset_index(drop=True)
    del df

    x_train = np.load(CACHE / "train.npz")["x"]
    model = classifier.train_inspection_model(
        x_train, train["label"].tolist(), CLASSES, seed=args.seed
    )
    # MD3-RS needs its own feature-bagged ensemble, trained on the same data
    # the inspection model saw. It is a detector, not a predictor: it never
    # supplies the class labels that form the monitored composition.
    rs = md3.train_random_subspace(
        x_train, train["label"].tolist(), CLASSES, seed=args.seed
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
    position = np.array([{l: i for i, l in enumerate(sequence)}[str(l)] for l in lots])

    x_clean = x_all[rows]
    off, warmup = args.phase_one, 20
    proba_clean, frame_clean = mc.build(model, x_clean, lots, sequence)
    coords_clean = composition.ilr_matrix(frame_clean)
    props_clean = composition.proportion_matrix(frame_clean)
    sizes = frame_clean["lot_size"].to_numpy()
    fit = monitor.calibrate(coords_clean[:off], sizes[:off], lam=0.2,
                            target_far=0.005, seed=args.seed)
    conf_clean = baselines.max_confidence_stream(proba_clean, lots, sequence)[off:]

    def scorers(coords, props, conf, proba=None, x=None):
        extra = {}
        if proba is not None and x is not None:
            extra = {
                "md3_conf": lambda: md3.md3_score(
                    md3.lot_margin_density(
                        md3.margin_indicator(proba), lots, sequence),
                    off, lam=0.2),
                "md3_rs": lambda: md3.md3_score(
                    md3.lot_margin_density(
                        md3.margin_indicator(rs.probabilities(x)), lots, sequence),
                    off, lam=0.2),
            }
        return {**extra, **{
            "ilr_mewma": lambda: monitor.run_monitor(
                fit, coords[off:], sizes[off:])["statistic"],
            "proportion_mewma": lambda: mc._proportion_statistic(props, sizes, off),
            "adwin_confidence": lambda: mc.score_adwin(conf),
            "ks_confidence": lambda: mc.score_ks(conf),
        }}

    clean_scores = {
        k: f() for k, f in
        scorers(coords_clean, props_clean, conf_clean, proba_clean, x_clean).items()
    }

    out = {"mode": args.mode, "confined": bool(args.confine_to_defective),
           "change_points": [], "runs": {}}

    for change_point in args.change_points:
        if time.time() - t0 > args.budget_seconds:
            print(f"budget reached after {len(out['change_points'])} change points",
                  flush=True)
            break
        post = position >= change_point
        x_deg = x_clean.copy()
        x_deg[post] = degraded_features(
            stream, rows, lots, sequence, post, args, change_point, proba_clean
        )
        proba_deg, frame_deg = mc.build(model, x_deg, lots, sequence)
        coords_deg = composition.ilr_matrix(frame_deg)
        props_deg = composition.proportion_matrix(frame_deg)
        conf_deg = baselines.max_confidence_stream(proba_deg, lots, sequence)[off:]
        cp = change_point - off

        print(f"\nchange point {change_point} (stream index {cp}) "
              f"[{time.time() - t0:.0f}s]", flush=True)
        for name, fn in scorers(
            coords_deg, props_deg, conf_deg, proba_deg, x_deg
        ).items():
            deg = fn()
            clean = clean_scores[name]
            for target in TARGETS_PER_1000:
                rate = target / 1000.0
                calib = clean[cp:]
                threshold = float(np.quantile(calib, 1.0 - rate, method="higher"))
                hits = np.flatnonzero(deg[cp:] > threshold)
                delay = float(hits[0]) if len(hits) else float("inf")
                fa = 1000.0 * int((deg[warmup:cp] > threshold).sum()) / max(cp - warmup, 1)
                out["runs"].setdefault(name, {}).setdefault(str(target), []).append(
                    {"change_point": change_point, "delay": delay, "fa_per_1000": fa}
                )
            row = out["runs"][name]
            parts = []
            for t in TARGETS_PER_1000:
                d = row[str(t)][-1]["delay"]
                parts.append(f"@{t:g}/1000: " + ("never" if not np.isfinite(d) else f"{d:.0f}"))
            print(f"  {name:<22s} " + "  ".join(parts), flush=True)
        out["change_points"].append(change_point)

    # --- summary ------------------------------------------------------------
    print("\n\nmedian delay (range) across "
          f"{len(out['change_points'])} change points: {out['change_points']}")
    header = "".join(f"{f'@{t:g}/1000':>22s}" for t in TARGETS_PER_1000)
    print(f"{'method':<22s}{header}")
    summary = {}
    for name, by_target in out["runs"].items():
        cells = []
        summary[name] = {}
        for target in TARGETS_PER_1000:
            delays = [r["delay"] for r in by_target[str(target)]]
            finite = [d for d in delays if np.isfinite(d)]
            n_never = len(delays) - len(finite)
            if not finite:
                cells.append(f"{'never (all)':>22s}")
                summary[name][str(target)] = {"median": None, "n_never": n_never}
                continue
            med = float(np.median(finite))
            text = f"{med:.0f} ({min(finite):.0f}-{max(finite):.0f})"
            if n_never:
                text += f" +{n_never}x never"
            cells.append(f"{text:>22s}")
            summary[name][str(target)] = {
                "median": med, "min": float(min(finite)), "max": float(max(finite)),
                "n_never": n_never, "n_runs": len(delays),
            }
        print(f"{name:<22s}" + "".join(cells))
    out["summary"] = summary

    RESULTS.mkdir(parents=True, exist_ok=True)
    confine = "_conf" if args.confine_to_defective else ""
    tag = f"_sev{args.severity}" if args.severity != 1.0 else ""
    path = RESULTS / f"replicate_{args.mode}{confine}{tag}.json"
    if path.exists():
        with open(path) as fh:
            previous = json.load(fh)
        if len(previous.get("change_points", [])) > len(out["change_points"]):
            print(f"\nNOT overwriting {path}: it holds "
                  f"{len(previous['change_points'])} change points, this run has "
                  f"{len(out['change_points'])}. A partial rerun must never "
                  f"replace a fuller result.")
            return
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
